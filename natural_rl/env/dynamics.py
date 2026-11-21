"""Arm-dynamics backend: joint TORQUE commands on the Kangaroo arm (fast, arm-only MuJoCo dynamics).

The command u in [-1, 1]^7 is a fraction of the registered joint torque limits (`configs/env/dynamics_v1.yaml`). A decision
integrates the scene's arm model for `ticks_per_decision` MuJoCo steps (gravity + Coriolis compensated by default, joint damping
and armature from the model, every other dof locked, joint speed limited to v_max like every other action mode). After the
decision the same two safety rules as the kinematic backend apply, on the executed result: joints beyond their limits are clipped
(and stopped, and counted as a proposed limit violation), and a whole-arm collision rejects the move (the arm stays at the
previous pose with zero velocity, counted as a proposed collision). Rewards, costs, observations and metrics are the shared code.

This is a different transition model from the kinematic backend (inertia and damping), so its results are a separate stratum
label (`backend = dynamics`), compared across action modes only through the shared, pooled HKS scalers.
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np

from ..registry import load_yaml
from .reach_env import ReachBench


class DynamicsBench(ReachBench):
    def __init__(self, scene, n_envs, profile, obs_spec, tasks, horizon: int = 100, dt: float = 0.05, seed: int = 0, config=None):
        super().__init__(scene, n_envs, profile, obs_spec, tasks, horizon=horizon, dt=dt, seed=seed)
        cfg = config or load_yaml('configs/env/dynamics_v1.yaml')
        self.cfg = cfg
        self.tau_max = np.asarray(cfg['torque_limit_nm'], dtype=np.float64)
        if self.tau_max.shape != (self.n_joints,):
            raise ValueError('torque_limit_nm needs one value per controlled joint')
        self.ticks = int(cfg['ticks_per_decision'])
        if abs(self.ticks * float(cfg['timestep_s']) - dt) > 1e-9:
            raise ValueError('ticks_per_decision x timestep_s must equal the decision interval')
        model = copy.deepcopy(scene.model)
        model.opt.timestep = float(cfg['timestep_s'])
        model.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
                                   | mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self.adr_v = np.asarray(scene.dof_adr)
        self.adr_q = np.asarray(scene.qpos_adr)
        self.others_v = np.setdiff1d(np.arange(model.nv), self.adr_v)
        self.others_q = np.setdiff1d(np.arange(model.nq), self.adr_q)
        model.dof_armature[self.others_v] = float(cfg['locked_armature'])
        self.model = model
        self.data = [mujoco.MjData(model) for _ in range(n_envs)]
        self.gravity_comp = bool(cfg['gravity_compensation'])
        self.rest = scene.rest_qpos.copy()

    # ------------------------------------------------------------------ state sync
    def _load(self, i, q, qd):
        d = self.data[i]
        d.qpos[:] = self.rest
        d.qvel[:] = 0.0
        d.qpos[self.adr_q] = q
        d.qvel[self.adr_v] = qd
        d.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, d)               # refreshes qfrc_bias used by the compensation

    def reset(self, task_index=None):
        obs = super().reset(task_index)
        for i in range(self.n):
            self._load(i, self.q[i], np.zeros(self.n_joints))
        return obs

    # ------------------------------------------------------------------ one decision
    def _transition(self, v):
        q, p = self.q, self.p
        u = np.clip(v / self.obs_spec.v_max, -1.0, 1.0)
        v_limit = self.obs_spec.v_max
        q_new = np.empty_like(q)
        qd_new = np.empty_like(q)
        for i in range(self.n):
            d = self.data[i]
            tau = u[i] * self.tau_max
            for _ in range(self.ticks):
                d.qfrc_applied[self.adr_v] = tau + (d.qfrc_bias[self.adr_v] if self.gravity_comp else 0.0)
                mujoco.mj_step(self.model, d)
                d.qvel[self.adr_v] = np.clip(d.qvel[self.adr_v], -v_limit, v_limit)
            q_new[i], qd_new[i] = d.qpos[self.adr_q], d.qvel[self.adr_v]
        clip_proposed = ((q_new < self.lower) | (q_new > self.upper)).any(axis=-1)
        q_hat = np.clip(q_new, self.lower, self.upper)
        qd_hat = np.where(q_hat != q_new, 0.0, qd_new)                 # a joint stopped by its limit has no velocity
        p_hat = np.empty_like(p)
        collided = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            p_hat[i], collided[i] = self.kin[i].evaluate(q_hat[i])
        q_next = np.where(collided[:, None], q, q_hat)                 # whole-arm rejection: stay, at rest
        qd_next = np.where(collided[:, None], 0.0, qd_hat)
        p_next = np.where(collided[:, None], p, p_hat)
        for i in range(self.n):
            if collided[i] or clip_proposed[i]:
                self._load(i, q_next[i], qd_next[i])
            else:                                                       # keep the locked dofs at rest between decisions
                d = self.data[i]
                d.qpos[self.others_q] = self.rest[self.others_q]
                d.qvel[self.others_v] = 0.0
        return q_next, p_next, collided, clip_proposed, qd_next
