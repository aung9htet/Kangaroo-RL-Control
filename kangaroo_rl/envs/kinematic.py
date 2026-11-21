"""Batched model of the MDP: the exact transition rule on n copies of the Kangaroo arm.

Each copy has its own MjData of the kinematic model; a step is forward kinematics plus a collision
test (tens of microseconds), so millions of transitions are cheap. Episodes are synchronous: every
copy starts and ends together, which keeps hindsight relabelling simple.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mdp import Reward, TransitionRule, VelocityBins, step_model
from ..scene import ArmKinematics, Scene


@dataclass
class Step:
    q_next: np.ndarray       # (n, A) next joint angles of the active joints
    collided: np.ndarray     # (n,) move rejected
    point_next: np.ndarray   # (n, 3) reach point after the step (base frame)
    point: np.ndarray        # (n, 3) reach point before the step
    reward: np.ndarray       # (n,)
    distance: np.ndarray     # (n,) |p(q') - g|


class KinematicReachEnv:
    """n parallel arms. `active` selects the controlled joints (default all 7); the others stay at home.
    `limits` optionally replaces the joint limits of the active joints (used by the 2-joint lattice)."""

    def __init__(self, scene: Scene, n_envs: int, bins: VelocityBins, reward: Reward, horizon: int,
                 check_collision=True, q_noise=0.0, active=None, limits=None, seed=0):
        self.scene = scene
        self.n = n_envs
        self.bins = bins
        self.reward = reward
        self.horizon = horizon
        self.check_collision = check_collision
        self.q_noise = q_noise
        self.active = np.arange(scene.n_joints) if active is None else np.asarray(active)
        lower, upper = scene.lower.copy(), scene.upper.copy()
        if limits is not None:
            lower[self.active], upper[self.active] = np.asarray(limits, dtype=np.float64).T
        self.lower, self.upper = lower, upper
        self.rule = TransitionRule(bins, lower, upper)
        self.kinematics = [ArmKinematics(scene) for _ in range(n_envs)]
        self.rng = np.random.default_rng(seed)
        self.q = np.tile(scene.home, (n_envs, 1))
        self.goal = np.zeros((n_envs, 3))
        self.point = np.zeros((n_envs, 3))
        self.t = 0

    @property
    def n_active(self) -> int:
        return len(self.active)

    def full(self, q_active):
        q = np.tile(self.scene.home, (len(q_active), 1))
        q[:, self.active] = q_active
        return q

    def full_actions(self, actions):
        actions = np.asarray(actions)
        if np.issubdtype(actions.dtype, np.floating):      # velocities from a continuous controller
            full = np.zeros((len(actions), self.scene.n_joints))
        else:
            full = np.full((len(actions), self.scene.n_joints), self.bins.zero, dtype=np.int64)
        full[:, self.active] = actions
        return full

    def reset_to(self, q_active, goals):
        """Start every copy at the given active-joint angles with the given goals."""
        self.q = self.full(np.asarray(q_active, dtype=np.float64))
        self.goal = np.asarray(goals, dtype=np.float64).copy()
        self.point = np.array([k.reach_point(q) for k, q in zip(self.kinematics, self.q)])
        self.t = 0
        return self.observe()

    def set_goals(self, goals):
        """New goals without moving the arms (retargeting); restarts the episode clock."""
        self.goal = np.asarray(goals, dtype=np.float64).copy()
        self.t = 0
        return self.observe()

    def observe(self):
        return self.q[:, self.active].copy(), self.goal.copy()

    def distance(self):
        return np.linalg.norm(self.point - self.goal, axis=-1)

    def step(self, actions):
        """Apply bin indices (n, A) -- or velocities (n, A) as floats -- and return the Step and whether
        the episode horizon is reached."""
        full = self.full_actions(np.asarray(actions))
        q_next = np.empty_like(self.q)
        collided = np.zeros(self.n, dtype=bool)
        point = np.empty((self.n, 3))
        for i in range(self.n):
            noise = None
            if self.q_noise > 0:
                noise = np.zeros(self.scene.n_joints)
                noise[self.active] = self.rng.normal(0.0, self.q_noise, self.n_active)
            q_next[i], collided[i], point[i] = step_model(self.rule, self.kinematics[i], self.q[i], full[i],
                                                          self.check_collision, noise)
        before = self.point.copy()
        velocity = self.rule.velocity(np.asarray(actions))
        reward = self.reward(point, self.goal, collided, velocity, before)
        self.q, self.point = q_next, point
        self.t += 1
        step = Step(q_next[:, self.active].copy(), collided, point.copy(), before, reward, self.distance())
        return step, self.t >= self.horizon
