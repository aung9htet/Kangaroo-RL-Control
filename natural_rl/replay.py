"""Goal-relabelling episode replay with history-aware rewards and observations (all off-policy controllers).

Transitions are stored as raw executed values per episode (positions, joints, commanded velocities, actions,
collision and clip flags) with ONE goal per episode; rewards and observations are never stored. A sample gets a
hindsight goal with probability `her_ratio` (the "future" strategy: an accepted end-effector position reached at a
uniformly chosen later step of the same episode). Relabelling changes the goal for the whole episode history, so
the in-tolerance counter, the "first hold" flag and every reward term are recomputed from step 0 against the new
goal; the n-step return and the next observation are built from those recomputed histories. This is valid because
the transition (q', collision) does not depend on the goal, only the reward does.

The time limit is a truncation, not a terminal state: bootstrapping continues from the true final state, so the
buffer never reports `done`.
"""
from __future__ import annotations

import numpy as np

from .rewards.profiles import RewardProfile
from .rewards.terms import history_terms
from .env.contract import ObsSpec


def _shift_back(array, fill):
    """array[:, t] -> value 'after decision t-1' at position t (fill at t = 0)."""
    first = np.full_like(array[:, :1], fill)
    return np.concatenate([first, array[:, :-1]], axis=1)


class GoalReplay:
    def __init__(self, capacity: int, horizon: int, n_joints: int, action_dim: int, action_dtype, profile: RewardProfile,
                 obs_spec: ObsSpec, lower, upper, her_ratio=0.8, n_step=1, gamma=0.95, seed=0):
        self.horizon, self.n_joints = horizon, n_joints
        self.episodes = max(1, capacity // horizon)
        self.profile, self.obs_spec = profile, obs_spec
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.her_ratio, self.n_step, self.gamma = her_ratio, n_step, gamma
        self.rng = np.random.default_rng(seed)
        e, h, j = self.episodes, horizon, n_joints
        self.q = np.zeros((e, h, j), dtype=np.float32)          # joints before the decision
        self.q_next = np.zeros((e, h, j), dtype=np.float32)     # accepted joints after it
        self.p = np.zeros((e, h, 3))                            # end effector before / after (float64: exact tolerance tests)
        self.p_next = np.zeros((e, h, 3))
        self.v_cmd = np.zeros((e, h, j), dtype=np.float32)      # commanded velocity, rad/s
        self.action = np.zeros((e, h, action_dim), dtype=action_dtype)
        self.collided = np.zeros((e, h), dtype=bool)
        self.clip = np.zeros((e, h), dtype=bool)
        self.goal = np.zeros((e, 3), dtype=np.float32)
        self.complete = np.zeros(e, dtype=bool)
        self.cursor = 0
        self.slots = np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.complete.sum()) * self.horizon

    # ------------------------------------------------------------------ writing (synchronous episodes)
    def begin(self, goals):
        n = len(goals)
        self.slots = (self.cursor + np.arange(n)) % self.episodes
        self.cursor = (self.cursor + n) % self.episodes
        self.complete[self.slots] = False
        self.goal[self.slots] = goals

    def add(self, t, q, q_next, p, p_next, v_cmd, action, collided, clip):
        s = self.slots
        self.q[s, t], self.q_next[s, t], self.p[s, t], self.p_next[s, t] = q, q_next, p, p_next
        self.v_cmd[s, t], self.action[s, t], self.collided[s, t], self.clip[s, t] = v_cmd, action, collided, clip

    def end(self):
        self.complete[self.slots] = True

    # ------------------------------------------------------------------ sampling
    def _history(self, e, goal):
        return history_terms(self.p[e], self.p_next[e], goal,
                             self.v_cmd[e].astype(np.float64), self.q_next[e].astype(np.float64), self.collided[e],
                             self.lower, self.upper, self.profile.config)

    def observations(self, e, t, goal, hist=None):
        """obs_v2 of episode e BEFORE decision t, recomputed for `goal` (count, held-once and previous command included)."""
        hist = self._history(e, goal) if hist is None else hist
        b = np.arange(len(e))
        q = np.where((t == 0)[:, None], self._q0(e), self.q_next[e, np.maximum(t - 1, 0)])
        count = _shift_back(hist['count'], 0)[b, t]
        held = _shift_back(hist['held_once'], False)[b, t]
        v_prev = hist['v_prev'][b, t]
        return self.obs_spec.encode(q, goal, v_prev, count, held)

    def _q0(self, e):
        return self.q[e, 0]

    def sample(self, batch: int) -> dict:
        complete = np.flatnonzero(self.complete)
        if len(complete) == 0:
            raise RuntimeError('no complete episode in the replay buffer yet')
        e = complete[self.rng.integers(len(complete), size=batch)]
        t = self.rng.integers(self.horizon, size=batch)
        goal = self.goal[e].astype(np.float64).copy()
        relabel = self.rng.random(batch) < self.her_ratio
        future = t + (self.rng.random(batch) * (self.horizon - t)).astype(np.int64)      # uniform in [t, H-1]
        goal[relabel] = self.p_next[e[relabel], future[relabel]]
        hist = self._history(e, goal)
        total, _ = self.profile.combine(hist)                                             # (B, H) rewards for this goal
        steps = np.minimum(self.n_step, self.horizon - t)
        ret = np.zeros(batch)
        cost = np.zeros(batch)
        b = np.arange(batch)
        unsafe = self.collided[e].astype(np.float64) + self.clip[e]
        for i in range(self.n_step):
            valid = i < steps
            ti = np.minimum(t + i, self.horizon - 1)
            ret += np.where(valid, self.gamma ** i * total[b, ti], 0.0)
            cost += np.where(valid, self.gamma ** i * unsafe[b, ti], 0.0)
        last = t + steps - 1                                                              # last decision of the segment
        q_end = self.q_next[e, last]
        # features after decision `last`: the command just issued becomes the next observation's previous command
        next_obs = self.obs_spec.encode(q_end, goal, self.v_cmd[e, last], hist['count'][b, last],
                                        hist['held_once'][b, last])
        return {'obs': self.observations(e, t, goal, hist), 'action': self.action[e, t], 'next_obs': next_obs,
                'reward': ret.astype(np.float32), 'cost': cost.astype(np.float32),
                'discount': (self.gamma ** steps).astype(np.float32), 'goal': goal.astype(np.float32),
                'relabelled': relabel, 'episode': e, 'step': t}
