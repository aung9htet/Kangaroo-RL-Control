"""Deterministic evaluation of a saved checkpoint over a task suite (kinematic backend).

Writes, per (backend, suite, checkpoint): `episode_metrics.parquet` (one row per task), `action_events.parquet`
(one row per decision: proposed vs accepted motion, flags, raw reward terms) and `trajectories.parquet` (one row per
frame). Metrics come from the executed accepted end-effector positions, never from commands. HKS and R-HKS are
computed only if the frozen scaler file exists and is locked; otherwise they are NaN with `hks_status =
scalers_not_locked`, and the raw descriptors are still stored so every score can be regenerated later.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .checkpoints import checkpoint_dir, read_index, step_name, tree_sha256
from .env.reach_env import ReachBench
from .env.tasks import load_manifest, task_arrays
from .metrics.hks import SCALED_METRICS, r_hks, score_arrays
from .metrics.kinematics import KinematicsConfig, savgol_derivatives, segment_kinematics
from .metrics.outcomes import episode_outcomes
from .models.registry import load_policy
from .registry import metrics_config, repo_root, suite_path
from .rollout import rollout

EVAL_SEED = 20261121
SCALER_PATH = 'configs/metrics/hks_scalers_v1.yaml'


def load_scalers(backend: str):
    """Locked HKS scalers for a backend, or None while they have not been calibrated and locked."""
    path = repo_root() / SCALER_PATH
    if not path.exists():
        return None
    doc = yaml.safe_load(path.read_text())
    if doc.get('status') != 'locked' or backend not in doc.get('backends', {}):
        return None
    return doc['backends'][backend]


def eval_dir(run_dir, backend: str, suite: str, step: int) -> Path:
    return Path(run_dir) / 'eval' / backend / suite / step_name(step)


def evaluate_checkpoint(run_dir, run_manifest: dict, scene, spec, profile, step: int, suite: str, backend: str = 'kinematic',
                        batch: int = 32, scalers='auto', max_tasks: int | None = None) -> dict:
    """Evaluate one checkpoint on every task of `suite` and write the three tables; returns the episode metrics frame."""
    if backend not in ('kinematic', 'physics'):
        raise NotImplementedError(f'unknown backend {backend!r}; the two strata are never silently substituted for each other')
    run_dir = Path(run_dir)
    ckpt = checkpoint_dir(run_dir, step)
    manifest = json.loads((ckpt / 'manifest.json').read_text())
    if tree_sha256(ckpt / 'policy_state') != manifest['sha256']:
        raise RuntimeError(f'checkpoint {step} files do not match their recorded SHA256')
    policy = load_policy(run_manifest['model'], ckpt / 'policy_state', spec)
    doc = load_manifest(suite_path(suite))
    tasks = task_arrays(doc)
    n_tasks = len(tasks['task_id']) if max_tasks is None else min(max_tasks, len(tasks['task_id']))
    cfg = KinematicsConfig.from_yaml(metrics_config())
    scalers = load_scalers(backend) if scalers == 'auto' else scalers
    horizon = int(doc['tasks'][0]['horizon_decisions'])
    if backend == 'physics':
        from .env.physics import PhysicsBench
        env = PhysicsBench(scene, profile, spec, tasks, horizon=horizon)
    else:
        env = ReachBench(scene, min(batch, n_tasks), profile, spec, tasks, horizon=horizon)
    rows, events, frames = [], [], []
    dt = env.dt
    for begin in range(0, n_tasks, env.n):
        index = np.arange(begin, min(begin + env.n, n_tasks))
        padded = np.r_[index, np.full(env.n - len(index), index[-1])]          # pad the last batch; padded rows are discarded
        roll = rollout(policy, env, padded, seed=EVAL_SEED + begin)
        for row_i, task_i in enumerate(index):
            task_id = tasks['task_id'][task_i]
            ee, q, goal = roll['ee'][row_i], roll['q'][row_i], roll['goal'][row_i]
            out = episode_outcomes(ee, goal, dt, cfg.tolerance)
            kin = segment_kinematics(ee, q, goal, out['terminal_hold'], cfg)
            rows.append({'task_id': task_id, 'split': suite, 'category': tasks['category'][task_i], 'run_id': run_manifest['run_id'],
                         'model': run_manifest['model'], 'reward': run_manifest['reward'], 'seed': run_manifest['seed'],
                         'checkpoint_step': int(step), 'checkpoint_sha256': manifest['sha256'], 'backend': backend,
                         **out, 'collision_proposals': int(roll['collision_proposed'][row_i].sum()),
                         'clip_proposals': int(roll['clip_proposed'][row_i].sum()),
                         'unsafe_events': float(roll['cost'][row_i].sum()), **kin})
            velocity = (savgol_derivatives(ee, dt)[0] if len(ee) >= cfg.sg_window else np.gradient(ee, dt, axis=0))
            for k in range(len(ee)):
                frames.append({'task_id': task_id, 'k': k, 't_s': k * dt, 'joint_angles_rad': list(q[k]),
                               'end_effector_xyz_m': list(ee[k]), 'goal_xyz_m': list(goal), 'velocity_m_s': list(velocity[k]),
                               'coordinate_frame': 'robot0_base_link', 'backend': backend})
            for k in range(roll['v_cmd'].shape[1]):
                events.append({'task_id': task_id, 'k': k + 1, 't_s': (k + 1) * dt, 'q_cmd_rad_s': list(roll['v_cmd'][row_i, k]),
                               'q_exec_rad_s': list(roll['v_exec'][row_i, k]),
                               'collision_proposed': bool(roll['collision_proposed'][row_i, k]),
                               'collision_rejected': bool(roll['collision_rejected'][row_i, k]),
                               'clip_proposed': bool(roll['clip_proposed'][row_i, k]), 'cost': float(roll['cost'][row_i, k]),
                               'raw_reward_terms': json.dumps({t: float(roll['raw'][t][row_i, k]) for t in roll['raw']}),
                               'weighted_reward_terms': json.dumps({t: float(roll['weighted'][t][row_i, k]) for t in roll['weighted']}),
                               'action': json.dumps([float(a) for a in roll['action'][row_i, k]])})
    metrics = pd.DataFrame(rows)
    metrics = score_frame(metrics, scalers)
    out_dir = eval_dir(run_dir, backend, suite, step)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(out_dir / 'episode_metrics.parquet', index=False)
    pd.DataFrame(events).to_parquet(out_dir / 'action_events.parquet', index=False)
    pd.DataFrame(frames).to_parquet(out_dir / 'trajectories.parquet', index=False)
    return metrics


def score_frame(metrics: pd.DataFrame, scalers) -> pd.DataFrame:
    """Add D1-D4, HKS and R_HKS (zero for failures and unscorable successes) when scalers are locked."""
    metrics = metrics.copy()
    if scalers is None:
        for column in ('D1_smooth', 'D2_timing', 'D3_spatial', 'D4_continuity', 'HKS', 'R_HKS'):
            metrics[column] = np.nan
        metrics['hks_status'] = 'scalers_not_locked'
        metrics['unscorable_success'] = False
        return metrics
    scored = score_arrays({m: metrics[m].to_numpy(dtype=float) for m in SCALED_METRICS}, scalers)
    for key, value in scored.items():
        metrics[key] = value
    valid = metrics['valid'].to_numpy(dtype=bool) & metrics['terminal_hold'].to_numpy(dtype=bool)
    r, unscorable = r_hks(metrics['terminal_hold'], metrics['HKS'], valid)
    metrics['R_HKS'], metrics['unscorable_success'], metrics['hks_status'] = r, unscorable, 'scored'
    return metrics


def select_checkpoint(run_dir, backend: str = 'kinematic', suite: str = 'val_v1') -> int:
    """Validation-selected checkpoint: highest terminal-hold rate, ties -> lower unsafe-event rate, then later step.

    Uses validation evaluations only (never test results); they must exist: `eval --suite val_v1 --all-checkpoints`.
    """
    base = Path(run_dir) / 'eval' / backend / suite
    scored = []
    for step in read_index(run_dir)['checkpoint_step'].astype(int):
        path = base / step_name(step) / 'episode_metrics.parquet'
        if not path.exists():
            raise FileNotFoundError(f'validation evaluation of checkpoint {step} is missing ({path}); run '
                                    f'`python -m natural_rl eval --run-id {Path(run_dir).name} --suite val_v1 --all-checkpoints` first. '
                                    'Test results are never used to choose a checkpoint.')
        df = pd.read_parquet(path)
        scored.append((df['terminal_hold'].mean(), -df['unsafe_events'].mean(), int(step)))
    return max(scored)[2]
