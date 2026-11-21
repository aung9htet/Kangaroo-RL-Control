"""The common action and observation contract shared by all ten controllers.

Action: every controller ends up commanding seven joint velocities in rad/s, bounded by +-v_max (1 rad/s,
a benchmark contract: the simulator itself enforces only joint position limits, see README.md, section 9).
  * MultiDiscrete(7,...,7) : bin index k_j -> v_max * LEVELS[k_j]   (the historical bins, unchanged)
  * Box[-1, 1]^7           : a_j -> v_max * clip(a_j, -1, 1)
Observation (fixed feature map, so every algorithm sees identical inputs):
  obs_v1 (historical parity only): phi(q) ++ phi(g)                              = 63 + 27 = 90 values
  obs_v2 (all benchmark runs)    : phi(q) ++ phi(g) ++ v_prev / v_max ++ min(count, 5)/5 ++ held_once  = 99 values
where phi is the sinusoidal positional encoding with L = 4 frequencies (kangaroo_rl.encoding), count is the
consecutive in-tolerance counter and held_once the "a 5-step hold has already completed" flag. The extra
obs_v2 inputs make the hold-aware rewards Markov; they are recomputed for relabelled goals by the replay buffer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kangaroo_rl.encoding import encoded_size, normalise, positional_encoding

LEVELS = (-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0)
N_BINS = len(LEVELS)


def multidiscrete_to_velocity(actions, v_max: float = 1.0) -> np.ndarray:
    """Bin indices (..., J) -> commanded velocities (..., J) in rad/s."""
    actions = np.asarray(actions)
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError('MultiDiscrete actions must be integer bin indices')
    if actions.min(initial=0) < 0 or actions.max(initial=0) >= N_BINS:
        raise ValueError(f'bin index outside 0..{N_BINS - 1}')
    return v_max * np.asarray(LEVELS)[actions]


def box_to_velocity(actions, v_max: float = 1.0) -> np.ndarray:
    """Normalised Box actions (..., J) -> commanded velocities (..., J) in rad/s; |v| <= v_max always."""
    return v_max * np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)


def box_was_clipped(actions) -> np.ndarray:
    """Per-decision flag: did the policy ask for more than the actuator limit (logged, separate from joint clipping)."""
    return (np.abs(np.asarray(actions)) > 1.0).any(axis=-1)


@dataclass(frozen=True)
class ObsSpec:
    version: str
    q_lower: np.ndarray
    q_upper: np.ndarray
    goal_lower: np.ndarray
    goal_upper: np.ndarray
    n_joints: int = 7
    joint_freqs: int = 4
    goal_freqs: int = 4
    hold_steps: int = 5
    v_max: float = 1.0

    def __post_init__(self):
        if self.version not in ('v1', 'v2'):
            raise ValueError(f'unknown observation version {self.version!r}')

    @property
    def extras(self) -> int:
        return self.n_joints + 2 if self.version == 'v2' else 0

    @property
    def dim(self) -> int:
        return encoded_size(self.n_joints, self.joint_freqs) + encoded_size(3, self.goal_freqs) + self.extras

    def encode(self, q, goal, v_prev=None, count=None, held_once=None) -> np.ndarray:
        """Flat float32 observation(s). q (..., J), goal (..., 3); v_prev (..., J), count/held_once (...) for v2."""
        q = np.asarray(q, dtype=np.float64)
        parts = [positional_encoding(normalise(q, self.q_lower, self.q_upper), self.joint_freqs),
                 positional_encoding(normalise(np.asarray(goal, dtype=np.float64), self.goal_lower, self.goal_upper),
                                     self.goal_freqs)]
        if self.version == 'v2':
            if v_prev is None or count is None or held_once is None:
                raise ValueError('obs_v2 needs v_prev, count and held_once')
            parts += [np.asarray(v_prev, dtype=np.float64) / self.v_max,
                      (np.minimum(np.asarray(count), self.hold_steps) / self.hold_steps)[..., None],
                      np.asarray(held_once, dtype=np.float64)[..., None]]
        return np.concatenate(parts, axis=-1).astype(np.float32)
