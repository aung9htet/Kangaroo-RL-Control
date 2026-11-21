"""Benchmark DecQN vs the frozen historical agent: same network function, same TD target, same schedules."""
import numpy as np
import pytest
import torch

from kangaroo_rl.agent.decqn import DecQNAgent
from kangaroo_rl.agent.network import DecQNetwork
from natural_rl.env.contract import ObsSpec
from natural_rl.models.decqn import BenchDecQNAgent, DecQNPolicy, FlatDecQNetwork, epsilon, policy_from_agent

J = 7
Q_LO, Q_HI = np.linspace(-1.5, -0.5, J), np.linspace(0.5, 1.5, J)
G_LO, G_HI = np.array([0.2, -0.6, 0.0]), np.array([0.7, 0.3, 0.4])


def make_pair(seed=0, hidden=32):
    torch.manual_seed(seed)
    network_cfg = dict(q_lower=Q_LO.tolist(), q_upper=Q_HI.tolist(), goal_lower=G_LO.tolist(), goal_upper=G_HI.tolist(),
                       n_bins=7, joint_freqs=4, goal_freqs=4, hidden=hidden, depth=2)
    historical = DecQNAgent(network_cfg, gamma=0.95, reward_bounds=(-2.01, 1.0))
    spec = ObsSpec('v1', Q_LO, Q_HI, G_LO, G_HI)
    bench = BenchDecQNAgent(spec.dim, (-2.01, 1.0), J, hidden, 2)
    for name in ('trunk', 'head'):                                           # identical architecture: copy the weights over
        getattr(bench.net, name).load_state_dict(getattr(historical.net, name).state_dict())
        getattr(bench.target, name).load_state_dict(getattr(historical.target, name).state_dict())
    return historical, bench, spec


def inputs(rng, spec, n=64):
    q = rng.uniform(Q_LO, Q_HI, (n, J))
    g = rng.uniform(G_LO, G_HI, (n, 3))
    return q, g, spec.encode(q, g)


def test_flat_network_equals_the_historical_network_on_obs_v1():
    historical, bench, spec = make_pair()
    q, g, obs = inputs(np.random.default_rng(1), spec)
    with torch.no_grad():
        want = historical.net(torch.as_tensor(q, dtype=torch.float32), torch.as_tensor(g, dtype=torch.float32))
        got = bench.net(torch.as_tensor(obs))
    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
    assert obs.shape[1] == 90                                                # 63 + 27 encoded features: the historical input


def test_double_q_td_target_and_n_step_discount_are_identical():
    historical, bench, spec = make_pair(seed=2)
    rng = np.random.default_rng(3)
    q, g, obs_next = inputs(rng, spec)
    reward = rng.normal(0, 0.5, len(q)).astype(np.float32)
    discount = np.where(rng.random(len(q)) < 0.5, 0.95 ** 3, 0.95 ** 2).astype(np.float32)
    want = historical.td_target(reward, q, g, discount)
    got = bench.td_target(reward, obs_next, discount)
    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


def test_targets_are_clipped_to_the_profile_bounds_over_one_minus_gamma():
    _, bench, spec = make_pair()
    _, _, obs = inputs(np.random.default_rng(4), spec, 8)
    y = bench.td_target(np.full(8, 1e6, dtype=np.float32), obs, np.full(8, 0.95, dtype=np.float32))
    assert float(y.max()) <= 1.0 / (1 - 0.95) + 1e-6
    y = bench.td_target(np.full(8, -1e6, dtype=np.float32), obs, np.full(8, 0.95, dtype=np.float32))
    assert float(y.min()) >= -2.01 / (1 - 0.95) - 1e-6


def test_update_fits_a_fixed_batch_and_polyak_moves_the_target():
    _, bench, spec = make_pair(seed=5)
    rng = np.random.default_rng(6)
    _, _, obs = inputs(rng, spec, 128)
    batch = {'obs': obs, 'next_obs': obs, 'action': rng.integers(0, 7, (128, J)), 'reward': rng.normal(0, 0.3, 128).astype(np.float32),
             'discount': np.full(128, 0.95 ** 3, dtype=np.float32)}
    first = bench.update(batch)['loss']
    before = [p.clone() for p in bench.target.parameters()]
    for _ in range(150):
        last = bench.update(batch)['loss']
    assert last < 0.4 * first
    assert any(not torch.equal(a, b) for a, b in zip(before, bench.target.parameters()))


def test_greedy_action_is_the_per_joint_argmax_of_the_utilities():
    _, bench, spec = make_pair(seed=7)
    _, _, obs = inputs(np.random.default_rng(8), spec, 16)
    with torch.no_grad():
        utilities = bench.net(torch.as_tensor(obs))
    np.testing.assert_array_equal(bench.act(obs), utilities.argmax(-1).numpy())


def test_epsilon_schedule_matches_the_historical_training_loop():
    from kangaroo_rl.train import epsilon as historical
    cfg = {'steps': 1_000_000, 'eps_fraction': 0.15, 'eps_start': 1.0, 'eps_end': 0.02}
    for step in (0, 10_000, 75_000, 150_000, 400_000, 1_000_000):
        assert epsilon(step, 1_000_000, 1.0, 0.02, 0.15) == pytest.approx(historical(step, cfg))


def test_saved_policy_reloads_to_identical_deterministic_actions(tmp_path):
    _, bench, spec = make_pair(seed=9)
    policy = policy_from_agent(bench)
    obs = inputs(np.random.default_rng(10), spec, 32)[2]
    policy.save(tmp_path)
    reloaded = DecQNPolicy.load(tmp_path, spec)
    np.testing.assert_array_equal(policy.act(obs), reloaded.act(obs))
    np.testing.assert_array_equal(reloaded.act(obs), reloaded.act(obs))
    assert reloaded.action_kind == 'multidiscrete'
    # the detached copy does not follow later training updates of the live agent
    bench.update({'obs': obs, 'next_obs': obs, 'action': np.zeros((32, J), dtype=np.int64), 'reward': np.ones(32, dtype=np.float32),
                  'discount': np.full(32, 0.95, dtype=np.float32)})
    np.testing.assert_array_equal(policy.act(obs), reloaded.act(obs))
