"""Record, verify and backfill the raw lossless video of every saved checkpoint (task val_001, camera canonical_v1).

The recorder RELOADS the checkpoint from disk into a separate evaluation environment (so a training process that
is still running cannot be affected, and a checkpoint that cannot be reloaded is caught immediately), replays the
canonical validation task deterministically, renders the accepted state at t = 0 and after every decision, and
writes `videos/step_XXXXXXXXX/val_001/{raw.mkv, preview.mp4, video_manifest.json, rollout.parquet}` plus a row in
`video_index.csv`. Failures are recorded (status `failed`) and re-raised by the caller; nothing is silently skipped
and an already verified video is never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .checkpoints import checkpoint_dir, read_index, set_video_status, step_name, tree_sha256
from .env.factory import make_env
from .models.registry import load_policy
from .rollout import rollout
from .video import file_sha256, verify_ffv1, write_ffv1, write_json, write_preview

VIDEO_INDEX_FIELDS = ['checkpoint_step', 'checkpoint_sha256', 'video_path', 'video_sha256', 'task_id', 'evaluation_seed',
                      'camera_id', 'width', 'height', 'pixel_format', 'codec', 'fps', 'frame_count', 'duration_s', 'status',
                      'outcome', 'verified', 'error']


def video_dir(run_dir, step: int, task_id: str) -> Path:
    return Path(run_dir) / 'videos' / step_name(step) / task_id


def _index_path(run_dir) -> Path:
    return Path(run_dir) / 'video_index.csv'


def read_video_index(run_dir) -> pd.DataFrame:
    path = _index_path(run_dir)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=VIDEO_INDEX_FIELDS)


def _upsert_video_index(run_dir, row: dict) -> None:
    frame = read_video_index(run_dir)
    frame = frame[frame['checkpoint_step'] != row['checkpoint_step']]
    new = pd.DataFrame([{k: row.get(k, '') for k in VIDEO_INDEX_FIELDS}])
    frame = new if frame.empty else pd.concat([frame, new], ignore_index=True)
    path = _index_path(run_dir)
    temporary = path.with_suffix('.csv.partial')
    frame.sort_values('checkpoint_step').to_csv(temporary, index=False, columns=VIDEO_INDEX_FIELDS)
    temporary.replace(path)


def _task_hash(task: dict) -> str:
    return hashlib.sha256(json.dumps({'q0': [round(v, 9) for v in task['q0']], 'goal': [round(v, 9) for v in task['goal']]},
                                     sort_keys=True).encode()).hexdigest()


class VideoRecorder:
    """Holds the heavy objects (scene renderer, single-task environment) so many checkpoints reuse them."""

    def __init__(self, run_dir, run_info: dict, scene, spec, profile, val_tasks: dict, video_cfg: dict, render_cfg: dict,
                 val_doc: dict):
        from .render import RawRenderer
        self.run_dir, self.run_info, self.spec, self.profile = Path(run_dir), run_info, spec, profile
        self.video_cfg, self.render_cfg, self.tasks = video_cfg, render_cfg, val_tasks
        self.task_id = video_cfg['task_id']
        self.index = val_tasks['task_id'].index(self.task_id)
        self.task = next(t for t in val_doc['tasks'] if t['task_id'] == self.task_id)
        self.env = make_env(scene, 1, profile, spec, val_tasks, horizon=int(self.task['horizon_decisions']))
        self.renderer = RawRenderer(scene, render_cfg)

    def record(self, step: int, force: bool = False) -> dict:
        """Record (or re-record) the video of checkpoint `step`; returns the video index row (status ok or failed)."""
        ckpt = checkpoint_dir(self.run_dir, step)
        manifest = json.loads((ckpt / 'manifest.json').read_text())
        out = video_dir(self.run_dir, step, self.task_id)
        existing = read_video_index(self.run_dir)
        done = existing[(existing['checkpoint_step'] == step) & (existing['status'] == 'ok')]
        if len(done) and not force and (out / 'raw.mkv').exists():
            return done.iloc[0].to_dict()                                   # a verified video is never silently overwritten
        row = {'checkpoint_step': int(step), 'checkpoint_sha256': manifest['sha256'], 'task_id': self.task_id,
               'evaluation_seed': self.video_cfg['evaluation_seed'], 'camera_id': self.render_cfg['camera_id'],
               'width': self.render_cfg['width'], 'height': self.render_cfg['height'], 'fps': self.video_cfg['fps'],
               'status': 'failed', 'verified': False}
        try:
            if tree_sha256(ckpt / 'policy_state') != manifest['sha256']:
                raise RuntimeError('checkpoint files do not match the SHA256 in their manifest')
            policy = load_policy(self.run_info['model'], ckpt / 'policy_state', self.spec)
            roll = rollout(policy, self.env, [self.index], seed=int(self.video_cfg['evaluation_seed']), renderer=self.renderer)
            frames = roll['frames']
            decisions = int(roll['v_cmd'].shape[1])
            if len(frames) != decisions + 1:
                raise RuntimeError(f'{len(frames)} frames for {decisions} decisions (expected decisions + 1)')
            if os.path.exists(out / 'raw.mkv'):
                (out / 'raw.mkv').unlink()                                  # only reached for missing/failed/forced entries
            record = write_ffv1(frames, out / 'raw.mkv', fps=int(self.video_cfg['fps']))
            check = verify_ffv1(out / 'raw.mkv', record['frame_sha256'], int(self.video_cfg['fps']), record['width'], record['height'])
            if not check['ok']:
                raise RuntimeError('raw.mkv failed readback verification: ' + '; '.join(check['problems']))
            preview = write_preview(frames, out / 'preview.mp4', fps=int(self.video_cfg['fps']))
            self._write_rollout(out / 'rollout.parquet', roll)
            ee, goal = roll['ee'][0], roll['goal'][0]
            distance = np.linalg.norm(ee - goal, axis=1)
            outcome = {'terminal_hold': bool((distance[-5:] <= 0.02).all()), 'final_error_m': float(distance[-1]),
                       'collision_proposed_count': int(roll['collision_proposed'].sum()),
                       'clip_proposed_count': int(roll['clip_proposed'].sum())}
            vm = {'run_id': self.run_info['run_id'], 'model': self.run_info['model'], 'reward': self.run_info['reward'],
                  'backend': self.run_info['backend'], 'seed': self.run_info['seed'], 'checkpoint_step': int(step),
                  'checkpoint_sha256': manifest['sha256'], 'task_id': self.task_id,
                  'evaluation_seed': self.video_cfg['evaluation_seed'], 'start_goal_sha256': _task_hash(self.task),
                  'camera_id': self.render_cfg['camera_id'], 'render_version': self.render_cfg['render_version'],
                  'mujoco_gl': os.environ.get('MUJOCO_GL'), 'width': record['width'], 'height': record['height'],
                  'pixel_format': record['pixel_format'], 'codec': record['codec'], 'fps': record['fps'],
                  'frame_count': record['frame_count'], 'decision_count': decisions, 'dt_s': float(roll['dt']),
                  'duration_s': float(decisions * roll['dt']), 'frame_timestamps_s': check['timestamps_s'],
                  'frame_sha256': record['frame_sha256'], 'raw_video_sha256': record['video_sha256'],
                  'raw_video_bytes': record['bytes'], 'uncompressed_bytes': record['raw_bytes'],
                  'preview_written': preview, 'rollout_outcome': outcome, 'verified': True,
                  'verified_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  'independent_decoder_checked': check['independent_decoder_checked'],
                  'note': 'preview.mp4 is a lossy viewing copy; raw.mkv (FFV1) is the record of source frames'}
            write_json(out / 'video_manifest.json', vm)
            row.update(video_path=str(out / 'raw.mkv'), video_sha256=record['video_sha256'], pixel_format=record['pixel_format'],
                       codec=record['codec'], frame_count=record['frame_count'], duration_s=vm['duration_s'], status='ok',
                       outcome=json.dumps(outcome), verified=True, error='')
        except Exception as error:                                          # recorded, then re-raised by the caller
            row.update(status='failed', verified=False, error=f'{type(error).__name__}: {error}',
                       outcome='')
            traceback.print_exc()
        _upsert_video_index(self.run_dir, row)
        set_video_status(self.run_dir, step, row['status'])
        return row

    def reproduce(self, step: int) -> dict:
        """Re-render checkpoint `step` in memory and compare every frame with the stored video manifest.

        Nothing is written: this is the independent deterministic re-recording check (same saved weights, same task,
        same seed, same camera must give the same pixels). Returns {'identical': bool, 'differing_frames': [...]}.
        """
        ckpt = checkpoint_dir(self.run_dir, step)
        vm_path = video_dir(self.run_dir, step, self.task_id) / 'video_manifest.json'
        if not vm_path.exists():
            raise FileNotFoundError(f'no recorded video for step {step}; run `video record` first')
        vm = json.loads(vm_path.read_text())
        policy = load_policy(self.run_info['model'], ckpt / 'policy_state', self.spec)
        roll = rollout(policy, self.env, [self.index], seed=int(self.video_cfg['evaluation_seed']), renderer=self.renderer)
        from .video import frame_hash
        hashes = [frame_hash(f) for f in roll['frames']]
        differing = [k for k, (a, b) in enumerate(zip(hashes, vm['frame_sha256'])) if a != b]
        return {'identical': not differing and len(hashes) == len(vm['frame_sha256']), 'frames': len(hashes),
                'expected_frames': len(vm['frame_sha256']), 'differing_frames': differing[:20]}

    def _write_rollout(self, path, roll) -> None:
        n = roll['v_cmd'].shape[1]
        k = np.arange(n + 1)
        frame = pd.DataFrame({
            'k': k, 't_s': k * roll['dt'], 'joint_angles_rad': list(roll['q'][0]), 'end_effector_xyz_m': list(roll['ee'][0]),
            'goal_xyz_m': [list(roll['goal'][0])] * (n + 1), 'coordinate_frame': 'robot0_base_link', 'backend': self.spec.backend})
        decisions = pd.DataFrame({
            'k': k[1:], 'q_cmd_rad_s': list(roll['v_cmd'][0]), 'q_exec_rad_s': list(roll['v_exec'][0]),
            'collision_proposed': roll['collision_proposed'][0], 'clip_proposed': roll['clip_proposed'][0],
            'reward': roll['reward'][0], 'raw_reward_terms': [json.dumps({t: float(roll['raw'][t][0, i]) for t in roll['raw']})
                                                              for i in range(n)]})
        pd.merge(frame, decisions, on='k', how='left').to_parquet(path, index=False)

    def close(self):
        self.renderer.close()


def verify_run(run_dir) -> list:
    """Problems found in the checkpoint/video archive of a run (empty list = every checkpoint has a verified raw video)."""
    run_dir = Path(run_dir)
    problems = []
    ckpts = read_index(run_dir)
    videos = read_video_index(run_dir)
    steps, vsteps = set(ckpts['checkpoint_step'].astype(int)), set(videos['checkpoint_step'].astype(int))
    for s in sorted(steps - vsteps):
        problems.append(f'step {s}: checkpoint has no video index entry')
    for s in sorted(vsteps - steps):
        problems.append(f'step {s}: video index entry has no checkpoint')
    for _, c in ckpts.iterrows():
        s = int(c['checkpoint_step'])
        manifest_path = Path(c['path']) / 'manifest.json'
        if not manifest_path.exists() or tree_sha256(Path(c['path']) / 'policy_state') != c['sha256']:
            problems.append(f'step {s}: checkpoint missing or its SHA256 changed')
            continue
        v = videos[videos['checkpoint_step'] == s]
        if v.empty:
            continue
        v = v.iloc[0]
        if v['status'] != 'ok':
            problems.append(f'step {s}: video status {v["status"]!r} {v["error"]}')
            continue
        out = Path(v['video_path']).parent
        try:
            vm = json.loads((out / 'video_manifest.json').read_text())
        except Exception as error:
            problems.append(f'step {s}: unreadable video_manifest.json ({error})')
            continue
        if vm['checkpoint_sha256'] != c['sha256']:
            problems.append(f'step {s}: video was recorded from a different checkpoint')
        raw = out / 'raw.mkv'
        if not raw.exists():
            problems.append(f'step {s}: raw.mkv is missing')
            continue
        if file_sha256(raw) != vm['raw_video_sha256']:
            problems.append(f'step {s}: raw.mkv SHA256 differs from its manifest')
            continue
        check = verify_ffv1(out / 'raw.mkv', vm['frame_sha256'], vm['fps'], vm['width'], vm['height'])
        if not check['ok']:
            problems.append(f'step {s}: ' + '; '.join(check['problems']))
        if check['frame_count'] != vm['decision_count'] + 1:
            problems.append(f'step {s}: {check["frame_count"]} frames for {vm["decision_count"]} decisions')
        if not (out / 'rollout.parquet').exists():
            problems.append(f'step {s}: rollout.parquet missing')
    return problems
