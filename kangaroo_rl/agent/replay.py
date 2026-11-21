"""Episode replay with hindsight goals.

Transitions are stored as raw values (q, a, q', c, p(q')) per episode, with one goal per episode.
Rewards are not stored: they are recomputed for whatever goal a sample is given. This is valid because
the transition (q', c) = T(q, a) is the same for every goal; only the reward depends on g.
"""
from __future__ import annotations

import numpy as np


class EpisodeReplay:
    def __init__(self, capacity: int, horizon: int, n_joints: int, reward, bins, her_ratio=0.8, seed=0,
                 n_step=1, gamma=0.95):
        self.horizon = horizon
        self.episodes = max(1, capacity // horizon)
        self.reward, self.bins, self.her_ratio = reward, bins, her_ratio
        self.n_step, self.gamma = n_step, gamma
        self.rng = np.random.default_rng(seed)
        e, h, j = self.episodes, horizon, n_joints
        self.q = np.zeros((e, h, j), dtype=np.float32)
        self.action = np.zeros((e, h, j), dtype=np.int8)
        self.q_next = np.zeros((e, h, j), dtype=np.float32)
        self.collided = np.zeros((e, h), dtype=bool)
        self.point_next = np.zeros((e, h, 3), dtype=np.float32)
        self.point = np.zeros((e, h, 3), dtype=np.float32)        # reach point before the step
        self.goal = np.zeros((e, 3), dtype=np.float32)
        self.complete = np.zeros(e, dtype=bool)
        self.cursor = 0
        self.slots = np.zeros(0, dtype=np.int64)

    def __len__(self):
        return int(self.complete.sum()) * self.horizon

    def begin(self, goals):
        """Open one episode per environment with these goals."""
        n = len(goals)
        self.slots = (self.cursor + np.arange(n)) % self.episodes
        self.cursor = (self.cursor + n) % self.episodes
        self.complete[self.slots] = False
        self.goal[self.slots] = goals

    def add(self, t, q, action, q_next, collided, point_next, point=None):
        s = self.slots
        if point is not None:
            self.point[s, t] = point
        self.q[s, t], self.action[s, t], self.q_next[s, t] = q, action, q_next
        self.collided[s, t], self.point_next[s, t] = collided, point_next

    def end(self):
        self.complete[self.slots] = True

    def sample(self, batch: int) -> dict:
        """Uniform transitions from complete episodes; a fraction `her_ratio` gets the goal replaced by
        the reach point achieved at a uniformly chosen later step of the same episode ('future').

        With n_step > 1 the target uses the n-step return R = sum_i gamma^i r_{t+i} (truncated at the end of
        the episode, m <= n steps), every r_{t+i} recomputed for the sample's goal, and bootstraps from
        q_{t+m} with discount gamma^m."""
        episodes = np.flatnonzero(self.complete)
        e = episodes[self.rng.integers(len(episodes), size=batch)]
        t = self.rng.integers(self.horizon, size=batch)
        goal = self.goal[e].copy()
        hindsight = self.rng.random(batch) < self.her_ratio
        future = t + (self.rng.random(batch) * (self.horizon - t)).astype(np.int64)
        goal[hindsight] = self.point_next[e[hindsight], future[hindsight]]
        steps = np.minimum(self.n_step, self.horizon - t)
        ret = np.zeros(batch)
        for i in range(self.n_step):
            valid = i < steps
            ti = np.minimum(t + i, self.horizon - 1)
            action_i = self.action[e, ti].astype(np.int64)
            r = self.reward(self.point_next[e, ti], goal, self.collided[e, ti], self.bins.velocity(action_i),
                            self.point[e, ti])
            ret += np.where(valid, self.gamma ** i * r, 0.0)
        last = t + steps - 1
        return {'q': self.q[e, t], 'action': self.action[e, t].astype(np.int64), 'q_next': self.q_next[e, last],
                'goal': goal, 'reward': ret.astype(np.float32), 'discount': (self.gamma ** steps).astype(np.float32),
                'collided': self.collided[e, t]}
