"""What a run was made with: commits (and dirty state), image identity, library versions, seeds and file hashes."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from .registry import repo_root


def _git(repo, *args):
    try:
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(repo) -> dict:
    commit = _git(repo, 'rev-parse', 'HEAD')
    # Untracked source files count as dirty (a commit must contain the code that produced a run); paths ignored by
    # .gitignore (results/, data/, .runtime/, caches) do not.
    dirty_files = _git(repo, 'status', '--porcelain')
    return {'commit': commit, 'dirty': bool(dirty_files), 'dirty_files': (dirty_files or '').splitlines()[:50]}


def image_info() -> dict:
    fingerprint = Path('/opt/kangaroo-rl/image.fingerprint')
    return {'image_ref': os.environ.get('NATURAL_RL_IMAGE_REF'),
            'image_fingerprint': fingerprint.read_text().strip() if fingerprint.exists() else None,
            'base_image': 'aung9htet/kangaroo-libero:0.1.0-ubuntu-22.04'}


def versions() -> dict:
    import mujoco
    import numpy
    import torch
    out = {'python': platform.python_version(), 'numpy': numpy.__version__, 'torch': torch.__version__,
           'mujoco': mujoco.__version__, 'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for name in ('stable_baselines3', 'sb3_contrib', 'gymnasium', 'pandas', 'pyarrow', 'av', 'tensordict'):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def run_provenance() -> dict:
    root = repo_root()
    return {'rl': git_state(root), 'sim': git_state(root / 'sim' / 'Kangaroo_Libero'), **image_info(), 'versions': versions(),
            'host': platform.node(), 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'mujoco_gl': os.environ.get('MUJOCO_GL'), 'cpu_count': os.cpu_count()}


def disk_free_bytes(path) -> int:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free
