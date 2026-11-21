"""Raw reward terms f_i.

All terms are computed twice by design and checked against each other in the tests:

* `step_terms`    stateful, one decision at a time (live training and evaluation), O(1) per step;
* `history_terms` stateless, whole episode histories (B, H), used by hindsight relabelling, which must
                  recompute the dwell counter and the "first hold" flag against the relabelled goal.

Conventions (decision t produces accepted position p_next[t]; p_prev[t] is the position before it):
  d_t       = ||p_next[t] - goal||
  inside_t  = d_t <= tolerance
  count_t   = consecutive inside decisions ending at t (0 if outside); the initial state is not counted
  hold_t    = count_t >= hold_steps
  held_once_t = any hold_s for s <= t
Units: d, path in metres; velocities in rad/s; dt in seconds. `speed` and `accel` are dimensionless command
proxies, not energy, and `rom` is a range-exposure proxy, not joint stress.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TERM_NAMES = ('r0', 'hold', 'time', 'path', 'speed', 'accel', 'rom')


@dataclass(frozen=True)
class RewardConfig:
    dt: float = 0.05
    v_max: float = 1.0
    horizon_s: float = 5.0
    tolerance: float = 0.02
    hold_steps: int = 5
    l_ref: float = 0.5
    a_ref: float = 40.0
    rom_margin: float = 0.10
    d_far: float = 0.5
    sigma: float = 0.02
    lam_collision: float = 1.0
    lam_speed_r0: float = 0.01
    hold_inside: float = 0.2
    hold_dwell: float = 0.8
    max_ee_step: float = 0.15       # bound on end-effector travel per decision, used only for reward bounds

    @classmethod
    def from_yaml(cls, cfg: dict) -> 'RewardConfig':
        c = cfg['constants']
        w = cfg['terms']['hold']['weights']
        return cls(dt=float(c['dt_s']), v_max=float(c['v_max_rad_s']), horizon_s=float(c['episode_horizon_s']),
                   tolerance=float(c['tolerance_m']), hold_steps=int(c['hold_steps']), l_ref=float(c['l_ref_m']),
                   a_ref=float(c['a_ref_rad_s2']), rom_margin=float(c['rom_margin']),
                   hold_inside=float(w['inside']), hold_dwell=float(w['dwell']))


def initial_state(n: int, n_joints: int) -> dict:
    """Per-environment reward/observation state at the start of an episode."""
    return {'count': np.zeros(n, dtype=np.int64), 'held_once': np.zeros(n, dtype=bool),
            'v_prev': np.zeros((n, n_joints))}


def _r0(d, collided, v_cmd, cfg: RewardConfig):
    u = np.mean(np.square(v_cmd / cfg.v_max), axis=-1)
    return (-np.minimum(d / cfg.d_far, 1.0) + np.exp(-np.square(d) / (2.0 * cfg.sigma ** 2))
            - cfg.lam_collision * np.asarray(collided, dtype=np.float64) - cfg.lam_speed_r0 * u)


def _rom(q, q_lower, q_upper, cfg: RewardConfig):
    span = q_upper - q_lower
    margin = np.minimum(q - q_lower, q_upper - q) / span          # normalised distance to the nearest limit
    return -np.mean(np.square(np.maximum(0.0, (cfg.rom_margin - margin) / cfg.rom_margin)), axis=-1)


def _common(d, p_prev, p_next, v_cmd, v_prev, q_next, collided, count, held_once_before, q_lower, q_upper,
            cfg: RewardConfig) -> dict:
    inside = d <= cfg.tolerance
    hold = count >= cfg.hold_steps
    scale = cfg.dt / cfg.horizon_s
    return {
        'r0': _r0(d, collided, v_cmd, cfg),
        'hold': cfg.hold_inside * inside + cfg.hold_dwell * hold,
        'time': -scale * (~held_once_before),
        'path': -np.linalg.norm(p_next - p_prev, axis=-1) / cfg.l_ref,
        'speed': -scale * np.mean(np.square(v_cmd / cfg.v_max), axis=-1),
        'accel': -np.mean(np.square((v_cmd - v_prev) / (cfg.dt * cfg.a_ref)), axis=-1),
        'rom': _rom(q_next, q_lower, q_upper, cfg),
    }


def step_terms(state: dict, p_prev, p_next, goal, v_cmd, q_next, collided, q_lower, q_upper, cfg: RewardConfig):
    """One decision for n environments. Returns (raw terms dict, new state); `state` is not modified."""
    d = np.linalg.norm(p_next - goal, axis=-1)
    inside = d <= cfg.tolerance
    count = np.where(inside, state['count'] + 1, 0)
    terms = _common(d, p_prev, p_next, v_cmd, state['v_prev'], q_next, collided, count, state['held_once'],
                    q_lower, q_upper, cfg)
    new_state = {'count': count, 'held_once': state['held_once'] | (count >= cfg.hold_steps),
                 'v_prev': np.array(v_cmd, dtype=np.float64, copy=True)}
    terms['d'] = d
    return terms, new_state


def history_terms(p_prev, p_next, goal, v_cmd, q_next, collided, q_lower, q_upper, cfg: RewardConfig) -> dict:
    """Terms for whole episode histories. Arrays are (B, H, ...) ; goal is (B, 3) and may be a relabelled goal.

    Returns the raw terms (B, H) plus the exact observation features recomputed against `goal`:
    `count` and `held_once` after each decision (B, H) and `v_prev` before each decision (B, H, J).
    """
    d = np.linalg.norm(p_next - goal[:, None, :], axis=-1)
    inside = d <= cfg.tolerance
    index = np.arange(d.shape[1])[None, :]
    last_outside = np.maximum.accumulate(np.where(inside, -1, index), axis=1)
    count = index - last_outside                                   # consecutive inside ending at each decision
    count = np.where(inside, count, 0)
    held_after = np.maximum.accumulate(count >= cfg.hold_steps, axis=1)
    held_before = np.concatenate([np.zeros_like(held_after[:, :1]), held_after[:, :-1]], axis=1)
    v_prev = np.concatenate([np.zeros_like(v_cmd[:, :1]), v_cmd[:, :-1]], axis=1)
    terms = _common(d, p_prev, p_next, v_cmd, v_prev, q_next, collided, count, held_before, q_lower, q_upper, cfg)
    terms.update(d=d, count=count, held_once=held_after, v_prev=v_prev)
    return terms


def term_bounds(cfg: RewardConfig, n_joints: int = 7) -> dict:
    """(low, high) of each raw term over all states and actions, used for profile reward bounds."""
    scale = cfg.dt / cfg.horizon_s
    return {
        'r0': (-1.0 - cfg.lam_collision - cfg.lam_speed_r0, 1.0),
        'hold': (0.0, cfg.hold_inside + cfg.hold_dwell),
        'time': (-scale, 0.0),
        'path': (-cfg.max_ee_step / cfg.l_ref, 0.0),
        'speed': (-scale, 0.0),
        'accel': (-(2.0 / (cfg.dt * cfg.a_ref)) ** 2, 0.0),
        'rom': (-1.0, 0.0),
    }
