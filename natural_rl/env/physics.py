"""Physics-backed evaluation environment: the same decisions executed by Kangaroo_Libero (100 Hz PD control, 5 ticks per
decision) through the preserved `LiberoReachEnv`, with the policy reading the measured joint angles (mode B).

It exposes the interface of `ReachBench` (single environment) so rollouts, metrics and HKS are computed by exactly the same
code. Accepted end-effector positions are the MEASURED hand positions of the simulated robot; the whole-arm collision
rejection and joint clipping are the same rules as in the fast backend. It is an evaluation stratum only: training happens
on the fast backend, and the two backends are never pooled into one result table.
"""
from __future__ import annotations

import numpy as np

from ..costs import unsafe_cost
from ..rewards.profiles import RewardProfile
from ..rewards.terms import initial_state, step_terms
from .contract import ObsSpec


class PhysicsBench:
    n = 1

    def __init__(self, scene, profile: RewardProfile, obs_spec: ObsSpec, tasks: dict, horizon: int = 100, dt: float = 0.05,
                 mode: str = 'B'):
        from kangaroo_rl.envs.libero import LiberoReachEnv
        from kangaroo_rl.mdp import Reward, VelocityBins
        if obs_spec.action_mode == 'torque':
            raise NotImplementedError('the full-scene physics backend (KangarooEnv) accepts position targets only; torque commands are '
                                      'evaluated on the dynamics backend')
        if scene.env is None:
            raise RuntimeError('the physics backend needs the scene built with keep_env=True (the simulator environment)')
        self.scene, self.profile, self.obs_spec, self.tasks, self.horizon, self.dt = scene, profile, obs_spec, tasks, horizon, dt
        self.sim = LiberoReachEnv(scene, VelocityBins(v_max=obs_spec.v_max, dt=dt), Reward(), horizon, mode=mode, target_object=None,
                                   feedforward=obs_spec.action_mode == 'velocity')     # position steps are pure position targets
        self.lower, self.upper, self.n_joints = scene.lower.copy(), scene.upper.copy(), scene.n_joints
        self.q = np.tile(scene.home, (1, 1))
        self.p = np.zeros((1, 3))
        self.goal = np.zeros((1, 3))
        self.state = initial_state(1, self.n_joints)
        self.qd = np.zeros((1, self.n_joints))
        self.p_prev = np.zeros((1, 3))
        self.t = 0

    def reset(self, task_index):
        i = int(np.asarray(task_index).reshape(-1)[0])
        self.sim.reset(self.tasks['goal'][i], self.tasks['q0'][i])
        self.q = self.sim.measured()[None].copy()
        self.p = self.sim.reach_point()[None].copy()
        self.p_prev = self.p.copy()
        self.goal = np.asarray(self.tasks['goal'][i], dtype=np.float64)[None].copy()
        self.state = initial_state(1, self.n_joints)
        self.qd = np.zeros((1, self.n_joints))
        self.t = 0
        return self.observe()

    def observe(self):
        s = self.state
        return self.obs_spec.encode(self.q, self.goal, s['v_prev'], s['count'], s['held_once'], self.qd,
                                    p=self.p, p_prev=self.p_prev, step=self.t)

    def step(self, v_cmd):
        v = np.asarray(v_cmd, dtype=np.float64).reshape(1, -1)
        if np.abs(v).max(initial=0.0) > self.obs_spec.v_max + 1e-9:
            raise ValueError('commanded speed exceeds the actuator limit')
        q, p = self.q, self.p
        clip = ((q + self.dt * v < self.lower) | (q + self.dt * v > self.upper)).any(axis=-1)
        result, _ = self.sim.step(v[0])                              # 5 simulator ticks; collision rejection as in the model
        q_next = self.sim.measured()[None].copy()
        p_next = np.asarray(result.point_next, dtype=np.float64)[None].copy()
        collided = np.array([bool(result.collided)])
        raw, self.state = step_terms(self.state, p, p_next, self.goal, v, q_next, collided, self.lower, self.upper, self.profile.config)
        total, weighted = self.profile.combine(raw)
        cost = unsafe_cost(collided, clip)
        info = {'q': q.copy(), 'q_next': q_next, 'p': p.copy(), 'p_next': p_next, 'v_cmd': v.copy(), 'v_exec': (q_next - q) / self.dt, 'qd_next': (q_next - q) / self.dt,
                'collision_proposed': collided.copy(), 'collision_rejected': collided.copy(), 'clip_proposed': clip, 'cost': cost,
                'raw': raw, 'weighted': weighted, 'distance': raw['d'], 'count': self.state['count'].copy(),
                'held_once': self.state['held_once'].copy(), 'tracking_error_rad': float(result.tracking)}
        self.p_prev = p
        self.q, self.p, self.qd = q_next, p_next, (q_next - q) / self.dt
        self.t += 1
        return self.observe(), total, self.t >= self.horizon, info
