"""Uniform policy interface shared by training, checkpointed replay, evaluation and video recording.

A Policy maps observations (n, D) to NATIVE actions of its action space and is saved/loaded from a directory:
  * action_kind 'multidiscrete' : integer bin indices (n, J)        -> v = v_max * LEVELS[k]
  * action_kind 'box'           : normalised actions in [-1, 1] (n, J) -> v = v_max * clip(a)
`act(..., deterministic=True)` must be free of exploration noise and of any randomness that is not seeded by the
caller, so the same saved checkpoint always replays the same trajectory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..env.contract import box_to_velocity, multidiscrete_to_velocity


class Policy:
    action_kind: str = 'box'
    model_id: str = ''

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the start of every episode (stateful planners clear their state here)."""

    def seed(self, seed: int) -> None:
        """Seed any internal randomness (TD-MPC2's planner); deterministic policies ignore it."""

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path, spec) -> 'Policy':
        raise NotImplementedError

    def info(self) -> dict:
        """Per-step policy diagnostics for logs (e.g. Growing Q's active resolution)."""
        return {}


def to_velocity(policy_or_kind, actions, v_max: float = 1.0) -> np.ndarray:
    kind = policy_or_kind if isinstance(policy_or_kind, str) else policy_or_kind.action_kind
    if kind == 'multidiscrete':
        return multidiscrete_to_velocity(actions, v_max)
    if kind == 'box':
        return box_to_velocity(actions, v_max)
    raise ValueError(f'unknown action kind {kind!r}')
