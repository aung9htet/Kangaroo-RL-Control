import numpy as np
import torch

from kangaroo_rl.agent.decqn import DecQNAgent, Explorer, joint_q
from kangaroo_rl.agent.replay import EpisodeReplay
from kangaroo_rl.mdp import Reward, VelocityBins

BINS = VelocityBins()
NETWORK = dict(q_lower=[-1.0] * 7, q_upper=[1.0] * 7, goal_lower=[-1.0] * 3, goal_upper=[1.0] * 3, n_bins=7,
               joint_freqs=4, goal_freqs=4, hidden=32, depth=2)


def test_td_target_is_double_q_and_clipped(rng):
    torch.manual_seed(0)
    agent = DecQNAgent(NETWORK, gamma=0.95, reward_bounds=(-2.01, 1.0))
    q_next, goal = rng.uniform(-1, 1, (16, 7)), rng.uniform(-1, 1, (16, 3))
    reward = rng.uniform(-1, 1, 16)
    with torch.no_grad():
        for p in agent.target.parameters():        # make the target net differ from the online net
            p.add_(0.1 * torch.randn_like(p))
        online = agent.net(torch.tensor(q_next, dtype=torch.float32), torch.tensor(goal, dtype=torch.float32))
        target = agent.target(torch.tensor(q_next, dtype=torch.float32), torch.tensor(goal, dtype=torch.float32))
        expected = torch.tensor(reward, dtype=torch.float32) + 0.95 * joint_q(target, online.argmax(-1))
    torch.testing.assert_close(agent.td_target(reward, q_next, goal), expected.clamp(agent.y_min, agent.y_max))
    assert np.isclose(agent.y_min, -2.01 / 0.05) and np.isclose(agent.y_max, 1.0 / 0.05)


def test_update_reduces_loss_on_fixed_batch(rng):
    torch.manual_seed(0)
    agent = DecQNAgent(NETWORK, gamma=0.0, lr=1e-2, tau=1.0)       # gamma 0: target = reward
    batch = {'q': rng.uniform(-1, 1, (64, 7)), 'goal': rng.uniform(-1, 1, (64, 3)),
             'action': rng.integers(7, size=(64, 7)), 'q_next': rng.uniform(-1, 1, (64, 7)),
             'reward': rng.uniform(-1, 1, 64).astype(np.float32)}
    first = agent.update(batch)['loss']
    for _ in range(200):
        last = agent.update(batch)['loss']
    assert last < 0.2 * first


def test_polyak_moves_target_towards_online():
    agent = DecQNAgent(NETWORK, tau=0.5)
    with torch.no_grad():
        for p in agent.net.parameters():
            p.fill_(1.0)
        for p in agent.target.parameters():
            p.fill_(0.0)
    agent.polyak()
    assert all(torch.allclose(p, torch.full_like(p, 0.5)) for p in agent.target.parameters())


def test_explorer_repeats_random_actions():
    explorer = Explorer(n_envs=1, n_joints=7, n_bins=7, max_repeat=5, seed=1)
    greedy = np.full((1, 7), 3)
    assert (explorer(greedy, 0.0) == 3).all()                      # epsilon 0: greedy
    first = explorer(greedy, 1.0)                                  # every joint starts a random action
    assert explorer.remaining.max() <= 4
    ongoing = explorer.remaining > 0
    second = explorer(greedy, 0.0)
    assert (second[ongoing] == first[ongoing]).all()               # ongoing random actions continue
    assert (second[~ongoing] == 3).all()                           # finished ones return to greedy


def test_replay_her_recomputes_rewards(rng):
    reward = Reward()
    replay = EpisodeReplay(capacity=40, horizon=10, n_joints=7, reward=reward, bins=BINS, her_ratio=1.0, seed=0)
    goals = rng.uniform(-1, 1, (2, 3))
    replay.begin(goals)
    for t in range(10):
        replay.add(t, rng.uniform(-1, 1, (2, 7)), rng.integers(7, size=(2, 7)), rng.uniform(-1, 1, (2, 7)),
                   rng.random(2) < 0.2, rng.uniform(-1, 1, (2, 3)))
    replay.end()
    batch = replay.sample(256)
    # every hindsight goal is a point achieved in the same episode, at or after the sampled step
    achieved = replay.point_next[replay.complete].reshape(-1, 3)
    assert all(np.any(np.all(np.isclose(g, achieved), axis=1)) for g in batch['goal'])
    # rewards are recomputed for the relabelled goal
    for i in range(10):
        q_next = batch['q_next'][i]
        e, t = np.argwhere(np.all(np.isclose(replay.q_next, q_next), axis=-1))[0]
        expected = reward(replay.point_next[e, t], batch['goal'][i], replay.collided[e, t],
                          BINS.velocity(replay.action[e, t].astype(int)))
        assert np.isclose(batch['reward'][i], expected, atol=1e-6)


def test_replay_without_her_keeps_goals(rng):
    replay = EpisodeReplay(capacity=20, horizon=10, n_joints=7, reward=Reward(), bins=BINS, her_ratio=0.0)
    goals = rng.uniform(-1, 1, (2, 3)).astype(np.float32)
    replay.begin(goals)
    for t in range(10):
        replay.add(t, np.zeros((2, 7)), np.zeros((2, 7), int), np.zeros((2, 7)), np.zeros(2, bool), np.zeros((2, 3)))
    replay.end()
    batch = replay.sample(50)
    assert all(np.any(np.all(np.isclose(g, goals), axis=1)) for g in batch['goal'])


def test_n_step_return_and_truncation(rng):
    reward = Reward()
    replay = EpisodeReplay(capacity=10, horizon=10, n_joints=7, reward=reward, bins=BINS, her_ratio=0.0, seed=0,
                           n_step=3, gamma=0.9)
    replay.begin(rng.uniform(-1, 1, (1, 3)))
    points = rng.uniform(-1, 1, (11, 3))
    for t in range(10):
        replay.add(t, np.full((1, 7), t), rng.integers(7, size=(1, 7)), np.full((1, 7), t + 1), np.zeros(1, bool),
                   points[None, t + 1], points[None, t])
    replay.end()
    batch = replay.sample(500)
    for i in range(50):
        t = int(batch['q'][i, 0])
        m = min(3, 10 - t)
        expected = sum(0.9 ** k * reward(points[t + k + 1], batch['goal'][i], False,
                                         BINS.velocity(replay.action[0, t + k].astype(int)), points[t + k])
                       for k in range(m))
        assert np.isclose(batch['reward'][i], expected, atol=1e-5)
        assert batch['q_next'][i, 0] == t + m and np.isclose(batch['discount'][i], 0.9 ** m)


def test_sequential_stages_are_consistent(rng):
    from kangaroo_rl.agent.seqq import SequentialQAgent
    torch.manual_seed(0)
    agent = SequentialQAgent(NETWORK)
    q, goal = torch.tensor(rng.uniform(-1, 1, (8, 7)), dtype=torch.float32), torch.tensor(rng.uniform(-1, 1, (8, 3)), dtype=torch.float32)
    actions = torch.tensor(rng.integers(7, size=(8, 7)))
    with torch.no_grad():
        hidden = agent.net.trunk_features(q, goal)
        batch_path = agent.net.stages(q, goal, actions)
        for j in range(7):
            torch.testing.assert_close(agent.net.stage(hidden, actions, j), batch_path[:, j])
        # the greedy action is the argmax of each stage given the earlier greedy choices
        greedy, value = agent.net.greedy(q, goal)
        stages = agent.net.stages(q, goal, greedy)
        assert (stages.argmax(-1) == greedy).all()
        torch.testing.assert_close(value, stages[:, 0].max(-1).values)


def test_sequential_update_fits_targets(rng):
    from kangaroo_rl.agent.seqq import SequentialQAgent
    torch.manual_seed(0)
    agent = SequentialQAgent(NETWORK, gamma=0.0, lr=1e-2, tau=1.0)
    batch = {'q': rng.uniform(-1, 1, (64, 7)), 'goal': rng.uniform(-1, 1, (64, 3)),
             'action': rng.integers(7, size=(64, 7)), 'q_next': rng.uniform(-1, 1, (64, 7)),
             'reward': rng.uniform(-1, 1, 64).astype(np.float32)}
    first = agent.update(batch)['loss']
    for _ in range(300):
        last = agent.update(batch)['loss']
    assert last < 0.3 * first
