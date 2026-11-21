"""Registries: models, reward profiles, cost budgets, task suites, run locations. One place that reads configs/."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

from .models.registry import MODEL_IDS


def repo_root() -> Path:
    return Path(os.environ.get('KANGAROO_RL_REPO', Path(__file__).resolve().parents[1]))


def results_root(smoke: bool = False) -> Path:
    base = Path(os.environ.get('NATURAL_RL_RESULTS', repo_root() / 'results'))
    return base / 'smoke' if smoke else base


def load_yaml(relative: str) -> dict:
    return yaml.safe_load((repo_root() / relative).read_text())


def file_sha(relative: str) -> str:
    return hashlib.sha256((repo_root() / relative).read_bytes()).hexdigest()


def model_config(model_id: str) -> dict:
    if model_id not in MODEL_IDS:
        raise KeyError(f'unknown model {model_id!r}; registered: {list(MODEL_IDS)}')
    path = repo_root() / 'configs' / 'models' / f'{model_id}.yaml'
    if not path.exists():
        raise FileNotFoundError(f'{path} is missing: the roster is fixed, so the model config must exist (milestone M4)')
    return yaml.safe_load(path.read_text())


def reward_config() -> dict:
    return load_yaml('configs/rewards/reward_v1.yaml')


def budgets_config() -> dict:
    return load_yaml('configs/costs/budgets_v1.yaml')


def metrics_config() -> dict:
    return load_yaml('configs/metrics/metrics_v1.yaml')


SUITES = ('train_v1', 'val_v1', 'test_v1', 'ood_v1')


def suite_path(suite: str) -> Path:
    if suite not in SUITES:
        raise KeyError(f'unknown suite {suite!r}; registered: {list(SUITES)} (plus checkpoint_video_v1)')
    return repo_root() / 'configs' / 'eval' / f'{suite}.json'


def run_dir(backend: str, model: str, reward: str, run_id: str, smoke: bool = False) -> Path:
    return results_root(smoke) / 'runs' / backend / model / reward / run_id


def find_run(run_id: str) -> Path:
    """Locate a run directory by id under results/runs (then results/smoke/runs). Exactly one must match."""
    matches = []
    for smoke in (False, True):
        matches += list((results_root(smoke) / 'runs').glob(f'*/*/*/{run_id}'))
    matches = [m for m in matches if (m / 'manifest.json').exists()]
    if not matches:
        raise FileNotFoundError(f'no run {run_id!r} under {results_root()}/runs or {results_root(True)}/runs')
    if len(matches) > 1:
        raise RuntimeError(f'run id {run_id!r} is ambiguous: {[str(m) for m in matches]}')
    return matches[0]
