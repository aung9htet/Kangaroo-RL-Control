import itertools

import numpy as np
import torch

from kangaroo_rl.agent.decqn import joint_q
from kangaroo_rl.mdp import Reward, TransitionRule, VelocityBins

BINS = VelocityBins()


def test_bins_levels_and_zero():
    assert BINS.n_bins == 7 and BINS.levels[BINS.zero] == 0.0
    np.testing.assert_allclose(BINS.velocity([[0, 3, 6]]), [[-1.0, 0.0, 1.0]])
    # per-step joint displacement: 0, +-0.005, +-0.015, +-0.05 rad
    np.testing.assert_allclose(np.sort(np.abs(BINS.velocity(np.arange(7)) * BINS.dt))[::2], [0, 0.005, 0.015, 0.05])


def test_nearest_inverts_velocity():
    assert (BINS.nearest(BINS.velocity(np.arange(7))) == np.arange(7)).all()


def test_transition_integrates_and_clips():
    rule = TransitionRule(BINS, lower=np.full(7, -1.0), upper=np.full(7, 1.0))
    q = np.array([0.0, 0.99, -0.99, 0.5, 0.0, 0.0, 0.0])
    a = np.array([6, 6, 0, 4, 2, 3, 5])
    expected = np.clip(q + 0.05 * BINS.velocity(a), -1, 1)
    np.testing.assert_allclose(rule.propose(q, a), expected)
    assert rule.propose(q, a)[1] == 1.0 and rule.propose(q, a)[2] == -1.0     # clipped at the limits
    np.testing.assert_allclose(rule.propose(q, a), rule.propose(q, a))        # deterministic


def test_float_actions_are_velocities():
    rule = TransitionRule(BINS, lower=np.full(7, -1.0), upper=np.full(7, 1.0))
    v = np.full(7, 0.2)
    np.testing.assert_allclose(rule.propose(np.zeros(7), v), np.full(7, 0.01))


def test_reward_terms():
    reward = Reward()
    goal = np.zeros(3)
    at_goal = reward(np.zeros(3), goal, False, np.zeros(7))
    assert np.isclose(at_goal, 1.0)                                          # bonus 1, no distance cost
    far = reward(np.array([2.0, 0, 0]), goal, False, np.zeros(7))
    assert np.isclose(far, -1.0)                                             # linear term saturates
    rejected = reward(np.zeros(3), goal, True, np.zeros(7))
    assert np.isclose(at_goal - rejected, reward.lam_c)
    fast = reward(np.zeros(3), goal, False, np.ones(7))
    assert np.isclose(at_goal - fast, reward.lam_a)
    r_min, r_max = reward.bounds
    assert r_min <= reward(np.array([5.0, 0, 0]), goal, True, np.ones(7)) <= r_max


def test_reward_prefers_closer_points():
    reward = Reward()
    d = np.linspace(0, 0.49, 50)                                  # below d_far the reward strictly decreases
    values = reward(np.stack([d, 0 * d, 0 * d], 1), np.zeros(3), np.zeros(50, bool), np.zeros((50, 7)))
    assert np.all(np.diff(values) < 0)


def test_factored_argmax_equals_brute_force(rng):
    """With Q = mean_j U_j(a_j) the per-joint argmax is the argmax over all 7^7 joint actions."""
    for _ in range(3):
        utilities = rng.normal(size=(7, 7))
        per_joint = utilities.argmax(axis=1)
        grids = np.stack(np.meshgrid(*[np.arange(7)] * 7, indexing='ij'), -1).reshape(-1, 7)
        values = utilities[np.arange(7), grids].mean(axis=1)
        assert (grids[values.argmax()] == per_joint).all()
        assert np.isclose(values.max(), utilities.max(axis=1).mean())


def test_joint_q_gathers_and_averages():
    utilities = torch.arange(2 * 7 * 7, dtype=torch.float32).view(2, 7, 7)
    actions = torch.tensor([[0] * 7, list(range(7))])
    expected = torch.stack([utilities[b, torch.arange(7), actions[b]].mean() for b in range(2)])
    torch.testing.assert_close(joint_q(utilities, actions), expected)


def test_shaping_is_potential_based(rng):
    """The shaping term is gamma Phi(s') - Phi(s): its discounted sum along any path depends only on the
    end points, so it cannot change which policy is optimal (Ng, Harada and Russell 1999)."""
    plain, shaped = Reward(), Reward(shaping=0.05, gamma=0.95)
    goal = np.zeros(3)
    path = np.cumsum(rng.normal(0, 0.02, (31, 3)), axis=0) + 0.3
    total = 0.0
    for t in range(30):
        extra = shaped(path[t + 1], goal, False, np.zeros(7), path[t]) - plain(path[t + 1], goal, False, np.zeros(7))
        total += 0.95 ** t * extra
    phi = lambda p: -np.linalg.norm(p - goal) / 0.05
    assert np.isclose(total, 0.95 ** 30 * phi(path[30]) - phi(path[0]))
    low, high = shaped.bounds
    assert low < plain.bounds[0] and high > plain.bounds[1]
