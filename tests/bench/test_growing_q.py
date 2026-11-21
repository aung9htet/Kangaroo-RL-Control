"""Growing Q: nested active action sets enforced by masking in acting, in the TD target and in exploration."""
import numpy as np
import torch

from natural_rl.env.contract import LEVELS
from natural_rl.env.contract import ObsSpec
from natural_rl.models.decqn import STAGES, BenchDecQNAgent, DecQNPolicy, MaskedExplorer, policy_from_agent, stage_at
from natural_rl.models.growing_q import GrowingQPolicy

J = 7


def agent_and_obs(seed=0, n=64):
    torch.manual_seed(seed)
    spec = ObsSpec('v1', np.full(J, -1.0), np.full(J, 1.0), np.zeros(3), np.ones(3))
    agent = BenchDecQNAgent(spec.dim, (-2.01, 1.0), J, 32, 2)
    rng = np.random.default_rng(seed)
    obs = spec.encode(rng.uniform(-1, 1, (n, J)), rng.uniform(0, 1, (n, 3)))
    return agent, obs


def test_stages_are_nested_and_have_2_3_5_7_bins_of_the_historical_levels():
    assert [len(s) for s in STAGES] == [2, 3, 5, 7]
    for small, large in zip(STAGES[:-1], STAGES[1:]):
        assert set(small) < set(large)
    assert [LEVELS[i] for i in STAGES[0]] == [-1.0, 1.0]                       # bang-bang first
    assert [LEVELS[i] for i in STAGES[1]] == [-1.0, 0.0, 1.0]
    assert STAGES[3] == list(range(7))


def test_linear_schedule_grows_at_the_registered_fractions():
    total, fractions = 1000, [0.2, 0.4, 0.6]
    got = [stage_at(s, total, fractions) for s in (0, 199, 200, 399, 400, 599, 600, 1000)]
    assert got == [0, 0, 1, 1, 2, 2, 3, 3]


def test_greedy_action_stays_inside_the_active_set_even_when_an_inactive_bin_has_the_highest_utility():
    agent, obs = agent_and_obs()
    with torch.no_grad():
        utilities = agent.net(torch.as_tensor(obs))
    unmasked = utilities.argmax(-1).numpy()
    agent.set_active(STAGES[0])
    masked = agent.act(obs)
    assert set(np.unique(masked)) <= set(STAGES[0])
    assert not np.array_equal(masked, unmasked)                                # the mask really changed some choices
    agent.set_active(STAGES[3])
    np.testing.assert_array_equal(agent.act(obs), unmasked)                    # the full set recovers plain DecQN


def test_double_q_target_maximises_only_over_active_bins():
    agent, obs = agent_and_obs(seed=1)
    reward = np.zeros(len(obs), dtype=np.float32)
    discount = np.full(len(obs), 0.95, dtype=np.float32)
    agent.set_active(STAGES[1])
    got = agent.td_target(reward, obs, discount)
    with torch.no_grad():
        online, target = agent.net(torch.as_tensor(obs)), agent.target(torch.as_tensor(obs))
    mask = torch.zeros(7, dtype=torch.bool)
    mask[STAGES[1]] = True
    chosen = online.masked_fill(~mask, float('-inf')).argmax(-1)
    want = 0.95 * target.gather(-1, chosen.unsqueeze(-1)).squeeze(-1).mean(-1)
    torch.testing.assert_close(got, want.clamp(agent.y_min, agent.y_max))


def test_exploration_samples_only_active_bins():
    explorer = MaskedExplorer(8, J, 7, max_repeat=3, seed=0)
    explorer.set_choices(STAGES[1])
    seen = set()
    for _ in range(200):
        seen |= set(np.unique(explorer(np.zeros((8, J), dtype=np.int64) + 3, epsilon=1.0)))
    assert seen <= set(STAGES[1]) and len(seen) == 3
    explorer.set_choices(STAGES[3])
    seen = set()
    for _ in range(200):
        seen |= set(np.unique(explorer(np.zeros((8, J), dtype=np.int64), epsilon=1.0)))
    assert seen == set(range(7))


def test_active_set_survives_a_checkpoint_reload_and_is_reported_per_step(tmp_path):
    agent, obs = agent_and_obs(seed=2)
    agent.set_active(STAGES[2])
    policy = policy_from_agent(agent)
    policy.save(tmp_path)
    reloaded = GrowingQPolicy.load(tmp_path, None)
    np.testing.assert_array_equal(policy.act(obs), reloaded.act(obs))
    assert reloaded.info() == {'active_bins': 5} and set(np.unique(reloaded.act(obs))) <= set(STAGES[2])
    plain, _ = agent_and_obs(seed=3)                                           # a plain DecQN checkpoint has all 7 bins active
    plain_dir = tmp_path / 'plain'
    plain_dir.mkdir()
    policy_from_agent(plain).save(plain_dir)
    assert DecQNPolicy.load(plain_dir, None).info() == {'active_bins': 7}
