"""Training orchestration: run setup, the checkpoint/video clock, logs and the run manifest.

A run is COMPLETE only when every saved checkpoint has a verified raw video. A video failure marks the run
incomplete and exits non-zero at the end (`--strict-video` aborts immediately); the checkpoint itself is never lost.
Training is launched by the user from the container shell; nothing here starts on its own.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .checkpoint_video import VideoRecorder, read_video_index, verify_run
from .checkpoints import read_index, save_checkpoint, scheduled_steps, step_name
from .env.contract import ObsSpec
from .env.reach_env import ReachBench
from .env.tasks import load_manifest, task_arrays
from .models.registry import ModelNotImplemented, train_model
from .provenance import disk_free_bytes, run_provenance
from .registry import (budgets_config, file_sha, load_yaml, model_config, repo_root, results_root, reward_config,
                       run_dir, suite_path)
from .rewards import load_profile
from .video import write_json

EXPECTED_VIDEO_BYTES = 25_000_000          # conservative per-video estimate (one measured pilot: 16.6 MB); replaced by measurement


class DiskSpaceError(RuntimeError):
    pass


class Clock:
    """Decides when checkpoints (and their videos) are due: step 0, every `every` transitions, final, and manual."""

    def __init__(self, ctx: 'TrainContext'):
        self.ctx = ctx
        self.pending = scheduled_steps(ctx.total_steps, ctx.checkpoint_every) if ctx.checkpoint_every else [ctx.total_steps]
        self.saved = set()
        self.failures = []

    def _save(self, step, kind, save_fn):
        ctx = self.ctx
        if step in self.saved:
            return
        manifest = save_checkpoint(ctx.run_dir, step, save_fn, kind=kind,
                                   meta={'run_id': ctx.run_id, 'model': ctx.model, 'reward': ctx.reward, 'seed': ctx.seed,
                                         'transitions': int(step), 'wall_s': round(time.time() - ctx.start_time, 1)})
        self.saved.add(step)
        print(f'[checkpoint] step {step:>9d}  {kind:9s} sha256 {manifest["sha256"][:12]}', flush=True)
        if ctx.recorder is not None and ctx.record_every_checkpoint:
            row = ctx.recorder.record(step)
            if row['status'] != 'ok':
                self.failures.append((step, row['error']))
                print(f'[ERROR] checkpoint {step}: video capture FAILED: {row["error"]}', flush=True)
                if ctx.strict_video:
                    raise RuntimeError(f'video capture failed at step {step}: {row["error"]}')
            else:
                print(f'[video]      step {step:>9d}  {row["frame_count"]} frames  raw.mkv sha256 {str(row["video_sha256"])[:12]}', flush=True)
        ctx.flush_logs()

    def on_start(self, step, save_fn):
        if self.pending and self.pending[0] == step:
            self.pending.pop(0)
            self._save(step, 'initial', save_fn)

    def after_step(self, step, save_fn):
        while self.pending and step >= self.pending[0]:
            due = self.pending.pop(0)
            if step != due:
                raise RuntimeError(f'checkpoint {due} was skipped (reached {step}); transitions must be a multiple of n_envs')
            self._save(step, 'final' if due == self.ctx.total_steps else 'scheduled', save_fn)
        manual = self.ctx.run_dir / 'MANUAL_CHECKPOINT'
        if manual.exists():
            manual.unlink()
            self._save(step, 'manual', save_fn)

    def on_end(self, step, save_fn):
        if step not in self.saved:
            self._save(step, 'final', save_fn)


@dataclass
class TrainContext:
    model: str
    reward: str
    run_id: str
    seed: int
    total_steps: int
    checkpoint_every: int
    record_every_checkpoint: bool
    strict_video: bool
    run_dir: Path
    scene: object
    spec: ObsSpec
    profile: object
    tasks: dict
    hyper: dict
    n_envs: int
    horizon: int = 100
    threads: int = 4
    device: str = 'auto'
    recorder: object = None
    cost_budget: object = None
    start_time: float = field(default_factory=time.time)
    clock: Clock = None
    _updates: list = field(default_factory=list)
    _episodes: list = field(default_factory=list)

    def make_env(self, n: int, seed: int | None = None) -> ReachBench:
        return ReachBench(self.scene, n, self.profile, self.spec, self.tasks, horizon=self.horizon,
                          seed=self.seed if seed is None else seed)

    def log_updates(self, step, updates, metrics: dict):
        self._updates.append({'transitions': int(step), 'updates': int(updates), 'wall_s': round(time.time() - self.start_time, 2),
                              'transitions_per_s': step / max(time.time() - self.start_time, 1e-9), **metrics})

    def log_episodes(self, step, ep: dict):
        n = len(ep['return'])
        for i in range(n):
            row = {'transitions': int(step), 'task_index': int(ep['task_index'][i]), 'episode_return': float(ep['return'][i]),
                   'episodic_cost': float(ep['cost'][i]), 'collision_proposals': float(ep['collisions'][i]),
                   'clip_proposals': float(ep['clips'][i]), 'final_distance_m': float(ep['final_distance'][i])}
            row.update({f'weighted_sum_{k}': float(v[i]) for k, v in ep['terms'].items()})
            self._episodes.append(row)

    def flush_logs(self):
        if self._updates:
            pd.DataFrame(self._updates).to_parquet(self.run_dir / 'train_history.parquet', index=False)
        if self._episodes:
            pd.DataFrame(self._episodes).to_parquet(self.run_dir / 'train_episodes.parquet', index=False)


def disk_preflight(root: Path, n_checkpoints: int, margin: float = 1.15) -> None:
    need = int(n_checkpoints * EXPECTED_VIDEO_BYTES * margin)
    free = disk_free_bytes(root)
    if free < need:
        raise DiskSpaceError(f'DISK-SPACE ALERT: this run needs about {need / 1e9:.1f} GB for {n_checkpoints} raw videos '
                             f'but only {free / 1e9:.1f} GB are free under {root}. Nothing is deleted or downsampled; free space '
                             'or point RESULTS_DIR at a larger volume (see .env.example).')


def apply_overrides(cfg: dict, assignments) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in assignments or []:
        key, _, raw = item.partition('=')
        if not key or not _:
            raise ValueError(f'override must look like hyper.key=value: {item!r}')
        node = cfg
        *parents, leaf = key.split('.')
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def run_training(args) -> int:
    from kangaroo_rl.config import load_config
    from kangaroo_rl.experiment import make_experiment

    from .models.registry import ensure_implemented
    ensure_implemented(args.model)                                  # a registered-but-unwritten model fails first, loudly
    root = repo_root()
    smoke = bool(args.smoke)
    hist_run = args.contract == 'hist_v1'
    mcfg = apply_overrides(model_config(args.model), [] if hist_run else args.set)
    rcfg = reward_config()
    hist = args.contract == 'hist_v1'
    if hist and (args.model != 'decqn' or args.reward != 'r0_original'):
        raise ValueError('--contract hist_v1 is the historical DecQN reproduction: --model decqn --reward r0_original')
    profile = load_profile(rcfg, args.reward)                       # refuses calibrated profiles until the register is locked
    n_envs = int(mcfg.get('n_envs', 16))
    if args.steps % n_envs or (args.checkpoint_every and args.checkpoint_every % n_envs):
        raise ValueError(f'--steps ({args.steps}) and --checkpoint-every ({args.checkpoint_every}) must be multiples of '
                         f'n_envs = {n_envs} so that checkpoints land on exact transition counts')
    backend = 'kinematic'
    out = run_dir(backend, args.model, args.reward, args.run_id, smoke)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} already exists; run ids are never reused or overwritten')
    out.mkdir(parents=True, exist_ok=False)
    steps = scheduled_steps(args.steps, args.checkpoint_every) if args.checkpoint_every else [args.steps]
    disk_preflight(results_root(smoke), len(steps) if args.record_every_checkpoint else 0)

    provenance = run_provenance()
    if provenance['rl']['dirty'] and not (args.allow_dirty or smoke):
        raise RuntimeError('refusing to start a registered run on a dirty working tree: commit first, or pass '
                           '--allow-dirty (such a run is recorded as dirty and excluded from main_v1)\n'
                           + '\n'.join(provenance['rl']['dirty_files'][:10]))

    exp = make_experiment(load_config(root / 'configs' / 'reach.yaml'))
    scene = exp.scene
    obs_version = 'v1' if hist else 'v2'                            # the historical reproduction keeps the original q + g input
    spec = ObsSpec(obs_version, scene.lower, scene.upper, exp.pool.lower, exp.pool.upper)
    train_doc = load_manifest(suite_path(args.suite))
    manifest = {'run_id': args.run_id, 'model': args.model, 'reward': args.reward, 'contract': args.contract, 'obs_version': obs_version,
                'backend': backend, 'train_backend': 'kinematic', 'seed': args.seed, 'steps': args.steps, 'suite': args.suite,
                'suite_tasks_sha256': train_doc['tasks_sha256'], 'checkpoint_every': args.checkpoint_every,
                'record_every_checkpoint': bool(args.record_every_checkpoint), 'cost_budget_id': args.cost_budget_id,
                'smoke': smoke, 'allow_dirty': bool(args.allow_dirty), 'status': 'running', 'provenance': provenance,
                'config_hashes': {'model': file_sha(f'configs/models/{args.model}.yaml'),
                                  'reward': file_sha('configs/rewards/reward_v1.yaml'),
                                  'metrics': file_sha('configs/metrics/metrics_v1.yaml'),
                                  'train_manifest': train_doc['tasks_sha256']},
                'reward_register_status': rcfg.get('status')}
    write_json(out / 'manifest.json', manifest)
    (out / 'config_resolved.yaml').write_text(yaml.safe_dump({'model': mcfg, 'reward_profile': args.reward,
                                                              'reward_terms': list(profile.terms), 'reward_weights': profile.weights,
                                                              'cost_budget_id': args.cost_budget_id, 'seed': args.seed,
                                                              'steps': args.steps}, sort_keys=False))
    ctx = TrainContext(model=args.model, reward=args.reward, run_id=args.run_id, seed=args.seed, total_steps=args.steps,
                       checkpoint_every=args.checkpoint_every, record_every_checkpoint=bool(args.record_every_checkpoint),
                       strict_video=bool(args.strict_video), run_dir=out, scene=scene, spec=spec, profile=profile,
                       tasks=task_arrays(train_doc), hyper=mcfg['hyper'], n_envs=n_envs, threads=args.threads, device=args.device)
    if args.record_every_checkpoint:
        val_doc = load_manifest(suite_path('val_v1'))
        ctx.recorder = VideoRecorder(out, manifest, scene, spec, profile, task_arrays(val_doc), load_yaml('configs/eval/checkpoint_video_v1.yaml'),
                                     load_yaml('configs/render/canonical_v1.yaml'), val_doc)
    if args.model == 'sac_pidlag':
        from .costs import PIDLagrangian
        if args.cost_budget_id is None:
            raise ValueError('sac_pidlag needs --cost-budget-id (budget_moderate_v1 or budget_strict_v1)')
        ctx.cost_budget = PIDLagrangian.from_yaml(budgets_config(), args.cost_budget_id)
    ctx.clock = Clock(ctx)
    status, error = 'complete', None
    try:
        if hist:
            from . import hist as hist_module
            manifest['parity_config_differences'] = hist_module.train(ctx, args, args.set)
            manifest['parity_with_reach_yaml'] = not manifest['parity_config_differences']
        else:
            train_model(args.model, ctx)
    except ModelNotImplemented:
        raise
    except BaseException as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        ctx.flush_logs()
        problems = []
        if status == 'complete':
            problems = verify_run(out) if args.record_every_checkpoint else []
            if problems or ctx.clock.failures:
                status = 'incomplete'
        manifest.update(status=status, error=error, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                        wall_s=round(time.time() - ctx.start_time, 1), video_problems=problems, n_checkpoints=len(read_index(out)))
        write_json(out / 'manifest.json', manifest)
        if ctx.recorder is not None:
            ctx.recorder.close()
    print(f'[run] {args.run_id}: {status}; {len(read_index(out))} checkpoints; results at {out}', flush=True)
    for problem in manifest['video_problems']:
        print(f'[ERROR] {problem}', flush=True)
    return 0 if status == 'complete' else 1
