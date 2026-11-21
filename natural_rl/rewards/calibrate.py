"""Locked reward-coefficient calibration (protocol section 06).

    lambda_i = min(10, 0.10 * median(|episode r0 return|) / max(1e-4, P75(|episode raw term i|)))

computed on pilot rollouts of the TRAINING tasks only (never validation-chosen or test tasks), pooled over all pilot
policies, with identical data for every controller. The unweighted episodic sums of the raw terms, the SHA256 of that
table, the code hash and the resulting coefficients are written into configs/rewards/reward_v1.yaml, which is then
marked `locked`. A locked register is immutable: changing it means a new benchmark id, so `lock_register` refuses to
touch it again, and `load_profile` refuses every calibrated profile until it is locked.
"""
from __future__ import annotations

import hashlib
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

CALIBRATED_TERMS = ('time', 'path', 'speed', 'accel', 'rom')


class AlreadyLocked(RuntimeError):
    pass


def lambda_formula(r0_returns, term_sums: dict, cap: float = 10.0, scale_fraction: float = 0.10, floor: float = 1e-4) -> dict:
    """Coefficients from episodic r0 returns (n,) and episodic raw term sums {term: (n,)}."""
    r0 = np.asarray(r0_returns, dtype=np.float64)
    r0 = r0[np.isfinite(r0)]
    if len(r0) == 0:
        raise ValueError('no pilot episodes: calibration needs real rollouts')
    scale = scale_fraction * float(np.median(np.abs(r0)))
    out = {}
    for term, values in term_sums.items():
        v = np.abs(np.asarray(values, dtype=np.float64))
        v = v[np.isfinite(v)]
        out[term] = float(min(cap, scale / max(floor, float(np.percentile(v, 75)))))
    return out


def table_sha256(frame: pd.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def collect_pilot(runs: list, split: str = 'train_v1', steps: str = 'final') -> pd.DataFrame:
    """Episodic unweighted raw-term sums of deterministic pilot rollouts: one row per (run, checkpoint, task)."""
    import yaml
    from kangaroo_rl.config import load_config
    from kangaroo_rl.experiment import make_experiment

    from ..checkpoints import checkpoint_dir, read_index
    from ..env.contract import ObsSpec
    from ..env.reach_env import ReachBench
    from ..env.tasks import load_manifest, task_arrays
    from ..models.registry import load_policy
    from ..registry import find_run, repo_root, reward_config, suite_path
    from ..rollout import rollout
    from . import load_profile

    if split != 'train_v1':
        raise ValueError('calibration uses the training split only')
    exp = make_experiment(load_config(repo_root() / 'configs' / 'reach.yaml'))
    tasks = task_arrays(load_manifest(suite_path(split)))
    profile = load_profile(reward_config(), 'r7_balanced', allow_unlocked=True)    # r7 contains every raw term
    rows = []
    for run_id in runs:
        run = find_run(run_id)
        import json
        manifest = json.loads((run / 'manifest.json').read_text())
        spec = ObsSpec(manifest.get('obs_version', 'v2'), exp.scene.lower, exp.scene.upper, exp.pool.lower, exp.pool.upper)
        saved = [int(s) for s in read_index(run)['checkpoint_step']]
        chosen = saved if steps == 'all' else [max(saved)]
        for step in chosen:
            policy = load_policy(manifest['model'], checkpoint_dir(run, step) / 'policy_state', spec)
            env = ReachBench(exp.scene, 32, profile, spec, tasks, horizon=100)
            for begin in range(0, len(tasks['task_id']), env.n):
                index = np.arange(begin, min(begin + env.n, len(tasks['task_id'])))
                padded = np.r_[index, np.full(env.n - len(index), index[-1])]
                roll = rollout(policy, env, padded, seed=20261121 + begin)
                for i, task in enumerate(index):
                    row = {'run_id': run_id, 'checkpoint_step': step, 'task_id': tasks['task_id'][task]}
                    row.update({f'sum_{t}': float(roll['raw'][t][i].sum()) for t in ('r0', 'hold', *CALIBRATED_TERMS)})
                    rows.append(row)
    return pd.DataFrame(rows)


def lock_register(path, lambdas: dict, n_episodes: int, data_sha256: str, code_sha: str, source: str) -> None:
    """Write coefficients and provenance into the register text (comments preserved) and mark it locked."""
    path = Path(path)
    text = path.read_text()
    if re.search(r'^status:\s*locked', text, flags=re.M):
        raise AlreadyLocked(f'{path} is already locked and immutable; a change of rules creates a new benchmark id')
    missing = [t for t in CALIBRATED_TERMS if t not in lambdas]
    if missing:
        raise ValueError(f'no coefficient for {missing}')
    text = re.sub(r'^status:\s*unlocked.*$', 'status: locked                   # locked by `calibrate rewards`; immutable', text, flags=re.M)
    for key, value in {'n_episodes': n_episodes, 'sha256': data_sha256, 'code_sha': code_sha,
                       'locked_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}.items():
        text, n = re.subn(rf'^(    {key}:)\s*.*$', rf'\g<1> {value!r}' if isinstance(value, str) else rf'\g<1> {value}', text, flags=re.M)
        if n != 1:
            raise ValueError(f'register has no `{key}` slot under calibration.data')
    text = re.sub(r'^(    source:)\s*.*$', lambda m: f'{m.group(1)} {source!r}', text, count=1, flags=re.M)
    for term in CALIBRATED_TERMS:
        text, n = re.subn(rf'^(    {term}:)\s*null\s*$', rf'\g<1> {lambdas[term]:.10g}', text, flags=re.M)
        if n != 1:
            raise ValueError(f'register has no unlocked coefficient slot for {term}')
    path.write_text(text)
