import numpy as np

from kangaroo_rl.dp import Lattice, Tables, greedy, rollout, transition_tables, value_iteration
from kangaroo_rl.mdp import Reward, VelocityBins


def toy_lattice(n=41):
    """Two 'joints' that move a point in a plane directly (p = (q1, 0, q2)), one blocked cell."""
    spacing = 0.005
    axes = [spacing * np.arange(n), spacing * np.arange(n)]
    q1, q2 = np.meshgrid(*axes, indexing='ij')
    points = np.stack([q1, np.zeros_like(q1), q2], -1)
    collides = np.zeros((n, n), dtype=bool)
    collides[20, 5:35] = True                                   # a wall
    shifts = np.array([-10, -3, -1, 0, 1, 3, 10])
    return Lattice(axes, points, collides, shifts, np.array([0, 1]))


def test_value_iteration_satisfies_bellman_equation():
    lattice, bins, reward = toy_lattice(), VelocityBins(), Reward(n_joints=2)
    goal = lattice.points[35, 20]
    tables = transition_tables(lattice, bins, reward, goal)
    value, q, sweeps, _ = value_iteration(tables, gamma=0.95, tol=1e-9)
    backup = (tables.reward + 0.95 * value[tables.next_i, tables.next_j]).max(axis=(0, 1))
    assert np.max(np.abs(backup - value)) < 1e-7                # V = max_a [r + gamma V(s')]
    assert sweeps < 3000


def test_rejected_moves_stay_and_are_penalised():
    lattice, bins, reward = toy_lattice(), VelocityBins(), Reward(n_joints=2)
    tables = transition_tables(lattice, bins, reward, lattice.points[0, 0])
    a = list(bins.levels).index(1.0)                            # +10 steps on joint 1
    zero = bins.zero
    # from (10, 10), +10 on joint 1 lands on the wall cell (20, 10): rejected, stays
    assert tables.rejected[a, zero, 10, 10] and tables.next_i[a, zero, 10, 10] == 10
    # limits clip: from the last index, moving up stays at the last index (not rejected)
    assert tables.next_i[a, zero, 40, 0] == 40 and not tables.rejected[a, zero, 40, 0]


def test_optimal_policy_goes_around_the_wall():
    lattice, bins, reward = toy_lattice(), VelocityBins(), Reward(n_joints=2)
    goal = lattice.points[35, 20]
    tables = transition_tables(lattice, bins, reward, goal)
    value, q, _, _ = value_iteration(tables, gamma=0.95, tol=1e-9)
    final, _ = rollout(lattice, tables, greedy(q), np.array([[5, 20]]), goal, horizon=100, gamma=0.95)
    assert final[0] < 1e-9                                       # reaches the goal behind the wall
