"""Experiment configuration: YAML files, where a file may name a `base` file it extends."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict:
    """Read a config; `base: other.yaml` (relative to this file) is loaded first and overridden."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop('base', None)
    if base is not None:
        data = _merge(load_config(path.parent / base), data)
    return data


def override(config: dict, assignments: list[str]) -> dict:
    """Apply command-line overrides such as `train.steps=20000` (values parsed as YAML)."""
    config = copy.deepcopy(config)
    for assignment in assignments:
        key, _, raw = assignment.partition('=')
        if not key or not _:
            raise ValueError(f'Override must look like section.key=value: {assignment!r}')
        node = config
        *parents, leaf = key.split('.')
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = yaml.safe_load(raw)
    return config
