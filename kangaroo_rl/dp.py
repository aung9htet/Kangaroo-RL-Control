"""Exact solution of the Bellman equation on a 2-joint lattice, and a check of the learned Q against it.

Two joints move (shoulder pitch arm_right_1 and elbow arm_right_4), the other five stay at home. On a
lattice with spacing h = 0.005 rad every velocity bin moves a whole number of lattice steps per decision
(v_max * level * dt / h = 0, +-1, +-3, +-10), so the model transition maps lattice points to lattice points
and the MDP is finite. Value iteration then solves

    V(s) = max_a [ r(s, a, T(s, a)) + gamma V(T(s, a)) ]

exactly (to a tolerance), with the same reward, collision rejection and discount as the full problem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import Reward, VelocityBins
from .scene import ArmKinematics, Scene


@dataclass
class Lattice:
    axes: list                 # lattice values per active joint
    points: np.ndarray         # (N1, N2, 3) reach point (base frame)
    collides: np.ndarray       # (N1, N2) configuration in collision
    shifts: np.ndarray         # (K,) lattice steps per decision for each bin
    active: np.ndarray

    @property
    def shape(self):
        return self.collides.shape

    def q(self, i, j):
        return np.stack([self.axes[0][i], self.axes[1][j]], axis=-1)


def build_lattice(scene: Scene, active, limits, spacing, bins: VelocityBins) -> Lattice:
    axes = []
    for low, high in limits:
        count = int(round((high - low) / spacing)) + 1
        axes.append(low + spacing * np.arange(count))
    shifts = np.asarray(bins.levels) * bins.v_max * bins.dt / spacing
    if not np.allclose(shifts, np.round(shifts)):
        raise ValueError('Velocity bins do not move whole lattice steps')
    kinematics = ArmKinematics(scene)
    shape = (len(axes[0]), len(axes[1]))
    points = np.zeros(shape + (3,))
    collides = np.zeros(shape, dtype=bool)
    q = scene.home.copy()
    for i, a in enumerate(axes[0]):
        for j, b in enumerate(axes[1]):
            q[active[0]], q[active[1]] = a, b
            points[i, j], collides[i, j] = kinematics.evaluate(q)
    return Lattice(axes, points, collides, np.round(shifts).astype(int), np.asarray(active))


@dataclass
class Tables:
    """For every joint action (k1, k2): next indices, rejection flags and the reward array for one goal."""
    next_i: np.ndarray        # (K, K, N1, N2)
    next_j: np.ndarray
    rejected: np.ndarray
    reward: np.ndarray


def transition_tables(lattice: Lattice, bins: VelocityBins, reward: Reward, goal) -> Tables:
    n1, n2 = lattice.shape
    k = len(lattice.shifts)
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')
    next_i = np.zeros((k, k, n1, n2), dtype=np.int32)
    next_j = np.zeros_like(next_i)
    rejected = np.zeros((k, k, n1, n2), dtype=bool)
    rewards = np.zeros((k, k, n1, n2))
    velocity = bins.v_max * np.asarray(bins.levels)
    for a in range(k):
        for b in range(k):
            ti = np.clip(i + lattice.shifts[a], 0, n1 - 1)
            tj = np.clip(j + lattice.shifts[b], 0, n2 - 1)
            hit = lattice.collides[ti, tj]
            ti, tj = np.where(hit, i, ti), np.where(hit, j, tj)
            next_i[a, b], next_j[a, b], rejected[a, b] = ti, tj, hit
            v = np.broadcast_to(np.array([velocity[a], velocity[b]]), (n1, n2, 2))
            rewards[a, b] = reward(lattice.points[ti, tj], goal, hit, v, lattice.points)
    return Tables(next_i, next_j, rejected, rewards)


def value_iteration(tables: Tables, gamma, tol=1e-6, max_sweeps=3000, snapshots=()):
    """Returns V*, Q* (K, K, N1, N2), the number of sweeps, and greedy policies at the requested sweeps."""
    value = np.zeros(tables.reward.shape[2:])
    saved = {}
    for sweep in range(1, max_sweeps + 1):
        q = tables.reward + gamma * value[tables.next_i, tables.next_j]
        new = q.max(axis=(0, 1))
        delta = np.max(np.abs(new - value))
        value = new
        if sweep in snapshots:
            saved[sweep] = greedy(q)
        if delta < tol:
            break
    q = tables.reward + gamma * value[tables.next_i, tables.next_j]
    return value, q, sweep, saved


def greedy(q):
    """Greedy joint action per lattice state from Q (K, K, N1, N2) -> (N1, N2, 2)."""
    k = q.shape[0]
    flat = q.reshape(k * k, *q.shape[2:]).argmax(axis=0)
    return np.stack([flat // k, flat % k], axis=-1)


def rollout(lattice: Lattice, tables: Tables, policy, starts, goal, horizon, gamma):
    """Follow `policy` (N1, N2, 2) from start indices; returns final distances and discounted returns."""
    i, j = starts[:, 0].copy(), starts[:, 1].copy()
    returns = np.zeros(len(i))
    for t in range(horizon):
        a, b = policy[i, j, 0], policy[i, j, 1]
        returns += gamma ** t * tables.reward[a, b, i, j]
        i, j = tables.next_i[a, b, i, j], tables.next_j[a, b, i, j]
    return np.linalg.norm(lattice.points[i, j] - goal, axis=-1), returns


def agent_policy(agent, lattice: Lattice, goal, batch=8192):
    """The agent's greedy action and max-Q value at every lattice state."""
    n1, n2 = lattice.shape
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing='ij')
    q = lattice.q(i.ravel(), j.ravel())
    actions = np.zeros((len(q), 2), dtype=np.int64)
    values = np.zeros(len(q))
    for begin in range(0, len(q), batch):
        part = q[begin:begin + batch]
        actions[begin:begin + batch], values[begin:begin + batch] = agent.greedy_and_value(
            part, np.tile(goal, (len(part), 1)))
    return actions.reshape(n1, n2, 2), values.reshape(n1, n2)


def compare(agent, lattice, tables, value, q_star, goal, horizon, gamma, n_starts=500, seed=0, tolerance=0.01):
    """How close the agent's greedy policy is to the exact optimum for one goal."""
    free = ~lattice.collides
    actions, learned = agent_policy(agent, lattice, goal)
    rows, cols = np.indices(lattice.shape)
    chosen = q_star[actions[..., 0], actions[..., 1], rows, cols]
    best = q_star.max(axis=(0, 1))
    delta = tolerance * (best[free].max() - best[free].min())
    exact = (actions == greedy(q_star)).all(axis=-1)
    rng = np.random.default_rng(seed)
    cells = np.argwhere(free)
    starts = cells[rng.choice(len(cells), n_starts, replace=False)]
    dp_final, dp_return = rollout(lattice, tables, greedy(q_star), starts, goal, horizon, gamma)
    rl_final, rl_return = rollout(lattice, tables, actions, starts, goal, horizon, gamma)
    return {
        'near_optimal_actions': float((chosen[free] >= best[free] - delta).mean()),
        'exact_actions': float(exact[free].mean()),
        'value_correlation': float(np.corrcoef(learned[free], value[free])[0, 1]),
        'value_mae': float(np.mean(np.abs(learned[free] - value[free]))),
        'dp_final_median': float(np.median(dp_final)), 'rl_final_median': float(np.median(rl_final)),
        'dp_final_p90': float(np.percentile(dp_final, 90)), 'rl_final_p90': float(np.percentile(rl_final, 90)),
        'regret_mean': float(np.mean(dp_return - rl_return)),     # >= 0; 0 means optimal returns
        'dp_return_mean': float(np.mean(dp_return)), 'rl_return_mean': float(np.mean(rl_return)),
        'tolerance': float(delta),
    }, actions, learned
