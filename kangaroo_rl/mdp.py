"""The Markov decision process, defined once and shared by the fast model, the simulator adapter,
the replay buffer (hindsight rewards), the dynamic-programming check and the baselines.

State     s = (q, g): 7 joint angles and the goal (reach-point target, robot base frame).
Action    a = (k_1..k_7): one velocity bin per joint, v_j = v_max * levels[k_j].
Transition (q', c) = T(q, a): q_hat = clip(q + dt * v, lower, upper); if q_hat is in collision the
          move is rejected (q' = q, c = 1), else q' = q_hat (c = 0). The goal never changes.
Reward    r = -min(d / d_far, 1) + exp(-d^2 / (2 sigma^2)) - lam_c * c - lam_a * |v|^2 / (n v_max^2),
          with d = |p(q') - g| the distance from the reach point to the goal.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scene import ArmKinematics, Scene


@dataclass(frozen=True)
class VelocityBins:
    """Discretised joint velocities: the same levels for every joint."""
    levels: tuple = (-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0)
    v_max: float = 1.0      # rad/s
    dt: float = 0.05        # s per decision

    @property
    def n_bins(self) -> int:
        return len(self.levels)

    @property
    def zero(self) -> int:
        return self.levels.index(0.0)

    def velocity(self, actions):
        """Bin indices (..., n_joints) -> joint velocities in rad/s."""
        return self.v_max * np.asarray(self.levels)[np.asarray(actions)]

    def nearest(self, velocities):
        """Velocity (..., n_joints) -> index of the closest bin (used by the quantised baseline)."""
        values = self.v_max * np.asarray(self.levels)
        return np.abs(np.asarray(velocities)[..., None] - values).argmin(axis=-1)


@dataclass(frozen=True)
class TransitionRule:
    """Integrate the commanded velocity over one decision period and clip to the joint limits.

    Collision rejection needs the kinematics and is applied by `step_model`.
    """
    bins: VelocityBins
    lower: np.ndarray
    upper: np.ndarray

    def velocity(self, actions):
        """Integer arrays are bin indices; float arrays are already velocities (continuous baselines)."""
        actions = np.asarray(actions)
        return actions if np.issubdtype(actions.dtype, np.floating) else self.bins.velocity(actions)

    def propose(self, q, actions):
        return np.clip(q + self.bins.dt * self.velocity(actions), self.lower, self.upper)


def step_model(rule: TransitionRule, kinematics: ArmKinematics, q, actions, check_collision=True,
               noise=None):
    """The full model transition for one arm: returns (q', c, p(q'))."""
    q_hat = rule.propose(q, actions)
    if noise is not None:
        q_hat = np.clip(q_hat + noise, rule.lower, rule.upper)
    point, collided = kinematics.evaluate(q_hat, check_collision)
    if collided:
        return np.array(q, dtype=np.float64), True, kinematics.reach_point(q)
    return q_hat, False, point


@dataclass(frozen=True)
class Reward:
    d_far: float = 0.5        # m; the linear term saturates at -1 beyond this distance
    sigma: float = 0.02       # m; width of the precision bonus near the goal
    lam_c: float = 1.0        # penalty for a rejected (colliding) move
    lam_a: float = 0.01       # penalty on commanded speed
    v_max: float = 1.0
    n_joints: int = 7
    shaping: float = 0.0      # m; delta of the potential Phi = -d / delta (0: no shaping term)
    gamma: float = 0.95       # discount used by the shaping term
    max_step: float = 0.15    # m; bound on how far the reach point moves in one decision
    max_distance: float = 1.5 # m; bound on the distance to a goal

    def __call__(self, point, goal, collided, velocity, point_before=None):
        """r(s, a, s'); with shaping, + gamma Phi(s') - Phi(s), which leaves the optimal policy unchanged."""
        distance = np.linalg.norm(np.asarray(point) - np.asarray(goal), axis=-1)
        speed = np.sum(np.square(velocity), axis=-1) / (self.n_joints * self.v_max ** 2)
        reward = (-np.minimum(distance / self.d_far, 1.0) + np.exp(-distance ** 2 / (2 * self.sigma ** 2))
                  - self.lam_c * np.asarray(collided, dtype=np.float64) - self.lam_a * speed)
        if self.shaping > 0:
            if point_before is None:
                raise ValueError('The shaping term needs the reach point before the step')
            before = np.linalg.norm(np.asarray(point_before) - np.asarray(goal), axis=-1)
            reward = reward + (before - self.gamma * distance) / self.shaping
        return reward

    @property
    def bounds(self):
        """(r_min, r_max) over all states and actions."""
        low, high = -1.0 - self.lam_c - self.lam_a, 1.0
        if self.shaping > 0:
            span = (self.max_step + (1 - self.gamma) * self.max_distance) / self.shaping
            low, high = low - span, high + span
        return low, high


@dataclass(frozen=True)
class GoalRegion:
    """Where objects can appear, relative to the table: metres beyond its near edge, sideways (base y),
    and above its top. Converted to a box in the robot base frame for a given scene."""
    x_from_edge: tuple = (0.03, 0.40)
    y: tuple = (-0.60, 0.30)
    z_above_top: tuple = (0.04, 0.30)

    def box(self, scene: Scene):
        edge, top = scene.table_edge_base, scene.table_top_base
        lower = np.array([edge + self.x_from_edge[0], self.y[0], top + self.z_above_top[0]])
        upper = np.array([edge + self.x_from_edge[1], self.y[1], top + self.z_above_top[1]])
        return lower, upper


class GoalPool:
    """Reach points the arm attains without collision inside the goal region, thinned to one point per
    voxel so that sampling from the pool is uniform over the reachable volume (not over joint space)."""

    def __init__(self, points, configurations, lower, upper):
        self.points = np.asarray(points, dtype=np.float64)
        self.configurations = np.asarray(configurations, dtype=np.float64)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)

    def __len__(self):
        return len(self.points)

    def sample(self, rng, n):
        index = rng.integers(len(self.points), size=n)
        return self.points[index].copy()

    @classmethod
    def build(cls, scene: Scene, region: GoalRegion, samples=2_000_000, voxel=0.02, seed=0, cache_dir=None,
              active=None, limits=None):
        """Sample joint angles uniformly (only the `active` joints, within `limits`, the rest at home),
        keep reach points inside the region that are collision-free, one per voxel."""
        lower, upper = region.box(scene)
        active = np.arange(scene.n_joints) if active is None else np.asarray(active)
        q_lower, q_upper = scene.lower.copy(), scene.upper.copy()
        if limits is not None:
            q_lower[active], q_upper[active] = np.asarray(limits, dtype=np.float64).T
        key = hashlib.sha1(f'{scene.signature()}|{lower.round(4)}|{upper.round(4)}|{samples}|{voxel}|{seed}'
                           f'|{active.tolist()}|{q_lower.round(4)}|{q_upper.round(4)}'.encode()).hexdigest()[:12]
        cache = Path(cache_dir) / f'goals-{key}.npz' if cache_dir else None
        if cache is not None and cache.exists():
            data = np.load(cache)
            return cls(data['points'], data['configurations'], lower, upper)
        kinematics = ArmKinematics(scene)
        rng = np.random.default_rng(seed)
        chosen, configurations, seen = [], [], set()
        q = scene.home.copy()
        for _ in range(samples):
            q[active] = rng.uniform(q_lower[active], q_upper[active])
            point = kinematics.reach_point(q)
            if np.any(point < lower) or np.any(point > upper):
                continue
            cell = tuple(np.floor((point - lower) / voxel).astype(int))
            if cell in seen:
                continue
            if kinematics.evaluate(q)[1]:
                continue
            seen.add(cell)
            chosen.append(point)
            configurations.append(q.copy())
        pool = cls(np.array(chosen), np.array(configurations), lower, upper)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, points=pool.points, configurations=pool.configurations)
        return pool


class StartSampler:
    """Start configurations: the home pose with noise, or a uniformly random collision-free pose.
    Only the `active` joints vary (within `limits`); the others stay at home. Returns active-joint angles."""

    def __init__(self, scene: Scene, home_noise=0.05, active=None, limits=None):
        self.scene = scene
        self.home_noise = home_noise
        self.active = np.arange(scene.n_joints) if active is None else np.asarray(active)
        self.lower, self.upper = scene.lower.copy(), scene.upper.copy()
        if limits is not None:
            self.lower[self.active], self.upper[self.active] = np.asarray(limits, dtype=np.float64).T
        self.kinematics = ArmKinematics(scene)

    def _free(self, draw, rng):
        for _ in range(1000):
            q = self.scene.home.copy()
            q[self.active] = draw(rng)
            if not self.kinematics.evaluate(q)[1]:
                return q[self.active]
        raise RuntimeError('No collision-free start found')

    def home(self, rng):
        a = self.active
        return self._free(lambda r: np.clip(self.scene.home[a] + r.normal(0.0, self.home_noise, len(a)),
                                            self.lower[a], self.upper[a]), rng)

    def random(self, rng):
        a = self.active
        return self._free(lambda r: r.uniform(self.lower[a], self.upper[a]), rng)
