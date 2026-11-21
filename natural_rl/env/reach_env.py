"""Benchmark reaching environment (fast "kinematic" backend) with full per-decision accounting.

The transition is the preserved historical rule (`kangaroo_rl.mdp.step_model`): propose q + dt v, clip to the joint
limits, reject the WHOLE move if the arm at the clipped pose collides. (Velocity and position-step commands are the same
transition here; `DynamicsBench` in `dynamics.py` overrides only `_transition` to integrate arm dynamics for torque commands.) What this wrapper adds, without changing
that rule, is the accounting the protocol requires at every decision: commanded vs executed velocity, proposed
collisions, pre-clip joint-limit proposals, the accepted end-effector position, and the raw and weighted reward
terms. Episodes are synchronous fixed blocks of `horizon` decisions (no terminal states; the time limit is a
truncation, so bootstrapping continues from the true final state).
"""
from __future__ import annotations

import numpy as np

from kangaroo_rl.scene import ArmKinematics

from ..costs import unsafe_cost
from ..rewards.profiles import RewardProfile
from ..rewards.terms import initial_state, step_terms
from .contract import ObsSpec


class ReachBench:
    def __init__(self, scene, n_envs: int, profile: RewardProfile, obs_spec: ObsSpec, tasks: dict, horizon: int = 100,
                 dt: float = 0.05, seed: int = 0):
        self.scene, self.n, self.profile, self.obs_spec = scene, n_envs, profile, obs_spec
        self.tasks, self.horizon, self.dt = tasks, horizon, dt
        self.lower, self.upper = scene.lower.copy(), scene.upper.copy()
        self.n_joints = scene.n_joints
        self.kin = [ArmKinematics(scene) for _ in range(n_envs)]
        self.rng = np.random.default_rng(seed)
        self.q = np.tile(scene.home, (n_envs, 1))
        self.p = np.zeros((n_envs, 3))
        self.goal = np.zeros((n_envs, 3))
        self.task_index = np.zeros(n_envs, dtype=np.int64)
        self.state = initial_state(n_envs, self.n_joints)
        self.qd = np.zeros((n_envs, self.n_joints))       # executed joint velocity (rad/s), the extra input of obs_v3
        self.p_prev = np.zeros((n_envs, 3))              # hand position one decision earlier (hand velocity feature)
        self.t = 0
        self._episode = None

    # ------------------------------------------------------------------ episodes
    def reset(self, task_index=None):
        """Start n episodes. `task_index` (n,) picks tasks; default draws uniformly with replacement (training)."""
        n_tasks = len(self.tasks['goal'])
        self.task_index = (self.rng.integers(n_tasks, size=self.n) if task_index is None
                           else np.asarray(task_index, dtype=np.int64))
        self.q = self.tasks['q0'][self.task_index].copy()
        self.goal = self.tasks['goal'][self.task_index].copy()
        self.p = np.array([k.reach_point(q) for k, q in zip(self.kin, self.q)])
        self.p_prev = self.p.copy()
        self.state = initial_state(self.n, self.n_joints)
        self.qd = np.zeros((self.n, self.n_joints))
        self.t = 0
        self._episode = {'return': np.zeros(self.n), 'cost': np.zeros(self.n), 'collisions': np.zeros(self.n),
                         'clips': np.zeros(self.n), 'terms': {k: np.zeros(self.n) for k in self.profile.terms}}
        return self.observe()

    def observe(self) -> np.ndarray:
        s = self.state
        return self.obs_spec.encode(self.q, self.goal, s['v_prev'], s['count'], s['held_once'], self.qd,
                                    p=self.p, p_prev=self.p_prev, step=self.t)

    def distance(self) -> np.ndarray:
        return np.linalg.norm(self.p - self.goal, axis=-1)

    # ------------------------------------------------------------------ one decision
    def _transition(self, v):
        """Kinematic rule: (q', p', collided, clip_proposed, qdot') for the commanded velocities v (n, J)."""
        q, p = self.q, self.p
        proposed = q + self.dt * v
        clip_proposed = ((proposed < self.lower) | (proposed > self.upper)).any(axis=-1)
        q_hat = np.clip(proposed, self.lower, self.upper)
        p_hat = np.empty_like(p)
        collided = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            p_hat[i], collided[i] = self.kin[i].evaluate(q_hat[i])
        q_next = np.where(collided[:, None], q, q_hat)          # a colliding move is rejected as a whole
        p_next = np.where(collided[:, None], p, p_hat)
        return q_next, p_next, collided, clip_proposed, (q_next - q) / self.dt

    def step(self, v_cmd):
        """Apply the normalised command times v_max, (n, J) (rad/s for velocity mode). Returns (obs, reward, done, info)."""
        v = np.asarray(v_cmd, dtype=np.float64)
        if v.shape != (self.n, self.n_joints):
            raise ValueError(f'expected velocities {(self.n, self.n_joints)}, got {v.shape}')
        if np.abs(v).max(initial=0.0) > self.obs_spec.v_max + 1e-9:
            raise ValueError('commanded speed exceeds the actuator limit; adapters must clip before the environment')
        q, p = self.q, self.p
        q_next, p_next, collided, clip_proposed, qd_next = self._transition(v)

        raw, self.state = step_terms(self.state, p, p_next, self.goal, v, q_next, collided, self.lower, self.upper,
                                     self.profile.config)
        total, weighted = self.profile.combine(raw)
        cost = unsafe_cost(collided, clip_proposed)
        info = {'q': q.copy(), 'q_next': q_next.copy(), 'p': p.copy(), 'p_next': p_next.copy(), 'v_cmd': v.copy(),
                'v_exec': (q_next - q) / self.dt, 'qd_next': qd_next.copy(), 'collision_proposed': collided.copy(), 'collision_rejected': collided.copy(),
                'clip_proposed': clip_proposed, 'cost': cost, 'raw': raw, 'weighted': weighted, 'distance': raw['d'],
                'count': self.state['count'].copy(), 'held_once': self.state['held_once'].copy()}
        self.p_prev = p
        self.q, self.p, self.qd = q_next, p_next, qd_next
        self.t += 1
        ep = self._episode
        ep['return'] += total
        ep['cost'] += cost
        ep['collisions'] += collided
        ep['clips'] += clip_proposed
        for k in ep['terms']:
            ep['terms'][k] += weighted[k]
        done = self.t >= self.horizon
        if done:
            info['episode'] = {'return': ep['return'].copy(), 'cost': ep['cost'].copy(), 'collisions': ep['collisions'].copy(),
                               'clips': ep['clips'].copy(), 'terms': {k: v.copy() for k, v in ep['terms'].items()},
                               'final_distance': self.distance(), 'task_index': self.task_index.copy()}
        return self.observe(), total, done, info
