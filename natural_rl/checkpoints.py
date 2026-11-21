"""Atomic, hashed checkpoints and the checkpoint index of a run.

A checkpoint is a directory `checkpoints/step_000150000/` holding `policy_state/` (everything needed to act
deterministically) and `manifest.json`. It is written to a temporary directory, fsynced and renamed, so a crash
never leaves a half-written checkpoint under its final name. Its SHA256 covers every file of `policy_state/` in
sorted order (names and contents), and is recorded in the manifest and in `checkpoint_index.csv`.
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import time
from pathlib import Path

import pandas as pd

INDEX_FIELDS = ['checkpoint_step', 'path', 'sha256', 'saved_at', 'kind', 'video_status']


def step_name(step: int) -> str:
    return f'step_{int(step):09d}'


def tree_sha256(directory) -> str:
    digest = hashlib.sha256()
    directory = Path(directory)
    for path in sorted(p for p in directory.rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
    return digest.hexdigest()


def _fsync_tree(directory: Path) -> None:
    for path in directory.rglob('*'):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def save_checkpoint(run_dir, step: int, save_policy, kind: str = 'scheduled', meta: dict | None = None) -> dict:
    """Atomically save `save_policy(policy_state_dir)` as checkpoint `step`; returns its manifest."""
    from .video import write_json

    root = Path(run_dir) / 'checkpoints'
    root.mkdir(parents=True, exist_ok=True)
    final = root / step_name(step)
    if final.exists():
        raise FileExistsError(f'{final} already exists; checkpoints are never overwritten')
    staging = root / (step_name(step) + '.partial')
    if staging.exists():
        shutil.rmtree(staging)
    (staging / 'policy_state').mkdir(parents=True)
    save_policy(staging / 'policy_state')
    digest = tree_sha256(staging / 'policy_state')
    manifest = {'checkpoint_step': int(step), 'kind': kind, 'sha256': digest,
                'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **(meta or {})}
    write_json(staging / 'manifest.json', manifest)
    _fsync_tree(staging)
    staging.rename(final)
    _append_index(run_dir, {'checkpoint_step': int(step), 'path': str(final), 'sha256': digest,
                            'saved_at': manifest['saved_at'], 'kind': kind, 'video_status': 'pending'})
    return manifest


def index_path(run_dir) -> Path:
    return Path(run_dir) / 'checkpoint_index.csv'


def read_index(run_dir) -> pd.DataFrame:
    path = index_path(run_dir)
    if not path.exists():
        return pd.DataFrame(columns=INDEX_FIELDS)
    return pd.read_csv(path)


def _write_index(run_dir, frame: pd.DataFrame) -> None:
    path = index_path(run_dir)
    temporary = path.with_suffix('.csv.partial')
    frame.sort_values('checkpoint_step').to_csv(temporary, index=False, columns=INDEX_FIELDS, quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def _append_index(run_dir, row: dict) -> None:
    frame = read_index(run_dir)
    frame = frame[frame['checkpoint_step'] != row['checkpoint_step']]
    new = pd.DataFrame([row])
    _write_index(run_dir, new if frame.empty else pd.concat([frame, new], ignore_index=True))


def set_video_status(run_dir, step: int, status: str) -> None:
    frame = read_index(run_dir)
    if int(step) not in set(frame['checkpoint_step'].astype(int)):
        raise KeyError(f'checkpoint {step} is not in the index of {run_dir}')
    frame.loc[frame['checkpoint_step'] == int(step), 'video_status'] = status
    _write_index(run_dir, frame)


def checkpoint_dir(run_dir, step: int) -> Path:
    path = Path(run_dir) / 'checkpoints' / step_name(step)
    if not path.is_dir():
        known = ', '.join(str(s) for s in read_index(run_dir)['checkpoint_step'].astype(int))
        raise FileNotFoundError(f'no checkpoint at step {step} in {run_dir}; saved steps: {known or "none"}')
    return path


def verify_checkpoint(run_dir, step: int) -> bool:
    """True if the files on disk still hash to the SHA256 in the manifest."""
    import json
    path = checkpoint_dir(run_dir, step)
    return tree_sha256(path / 'policy_state') == json.loads((path / 'manifest.json').read_text())['sha256']


def scheduled_steps(total: int, every: int = 50000) -> list:
    """Step 0, every `every` transitions, and the final step even when it is off the cadence."""
    steps = list(range(0, total + 1, every))
    if steps[-1] != total:
        steps.append(total)
    return steps
