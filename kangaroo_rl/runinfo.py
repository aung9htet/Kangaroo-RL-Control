"""What a run was made with: commits, image, versions and the simulator settings actually used."""
from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path


def _git(repo, *args):
    try:
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _commit(repo):
    commit = _git(repo, 'rev-parse', 'HEAD')
    if commit is None:
        return None
    dirty = bool(_git(repo, 'status', '--porcelain', '--untracked-files=no'))
    return commit + ('+dirty' if dirty else '')


def run_metadata(scene=None) -> dict:
    import mujoco
    import numpy
    import torch

    import kangaroo_libero

    rl_repo = Path(os.environ.get('KANGAROO_RL_REPO', Path(__file__).resolve().parents[1]))
    sim_repo = Path(os.environ.get('KANGAROO_REPO', rl_repo / 'sim/Kangaroo_Libero'))
    info = {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'host': platform.node(),
        'rl_commit': _commit(rl_repo),
        'sim_commit': _commit(sim_repo),
        'image_id': os.environ.get('KANGAROO_RL_IMAGE_ID'),
        'kangaroo_libero': kangaroo_libero.__file__,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'mujoco': mujoco.__version__,
        'numpy': numpy.__version__,
    }
    if scene is not None:
        info.update(settings_path=scene.settings_path, settings=scene.settings, scene_signature=scene.signature(),
                    suite=scene.suite, task_id=scene.task_id)
    return info
