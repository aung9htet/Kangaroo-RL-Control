"""Goal replay: relabelled samples must equal an independent live rollout against the relabelled goal."""
import copy
import os

import numpy as np
import pytest
import yaml

from natural_rl.env.contract import ObsSpec
from natural_rl.replay import GoalReplay
from natural_rl.rewards import initial_state, load_profile, step_terms

from .conftest import ROOT

J, H = 7, 30
LOWER, UPPER = np.full(J, -1.0), np.full(J, 1.0)


@pytest.fixture(scope='module')
def profile():
    with open(os.path.join(ROOT, 'configs', 'rewards', 'reward_v1.yaml')) as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    cfg['status'] = 'locked'
    cfg['calibration']['lambda'] = {'time': 1.5, 'path': 2.0, 'speed': 3.0, 'accel': 4.0, 'rom': 5.0}
    return load_profile(cfg, 'r7_balanced')


@pytest.fixture(scope='module')
def spec():
    return ObsSpec('v2', LOWER, UPPER, np.full(3, -1.0), np.full(3, 1.0))


def synthetic_episode(rng):
    """Positions move in short bursts separated by stationary stretches, so relabelled goals produce real dwell."""
    p = np.zeros((H, 3))
    pos = rng.uniform(-0.5, 0.5, 3)
    p_prev = np.zeros((H, 3))
    stay = 0
    for t in range(H):
        p_prev[t] = pos
        if stay == 0 and rng.random() < 0.25:
            stay = rng.integers(6, 10)
        if stay > 0:
            stay -= 1
        else:
            pos = pos + rng.normal(0, 0.05, 3)
        p[t] = pos
    q_next = rng.uniform(-0.99, 0.99, (H, J))
    q = np.concatenate([rng.uniform(-0.99, 0.99, (1, J)), q_next[:-1]])
    v = rng.choice([-1, -0.3, -0.1, 0, 0.1, 0.3, 1.0], size=(H, J)).astype(np.float32)
    return {'q': q, 'q_next': q_next, 'p': p_prev, 'p_next': p, 'v': v, 'action': rng.integers(0, 7, (H, J)).astype(np.int8),
            'collided': rng.random(H) < 0.1, 'clip': rng.random(H) < 0.05, 'goal': rng.uniform(-0.5, 0.5, 3)}


def fill(buffer, episodes, n_envs=2):
    for start in range(0, len(episodes), n_envs):
        group = episodes[start:start + n_envs]
        buffer.begin(np.array([g['goal'] for g in group]))
        for t in range(H):
            stack = lambda key: np.array([g[key][t] for g in group])
            buffer.add(t, stack('q'), stack('q_next'), stack('p'), stack('p_next'), stack('v'), stack('action'),
                       stack('collided'), stack('clip'))
        buffer.end()


def live_rollout(ep, goal, profile, spec):
    """Independent stepwise simulation against `goal`: observation before each decision, reward of each decision,
    observation after each decision."""
    state = initial_state(1, J)
    obs_before, rewards, obs_after = [], [], []
    for t in range(H):
        obs_before.append(spec.encode(ep['q'][t][None], goal[None], state['v_prev'], state['count'], state['held_once'])[0])
        v = ep['v'][t][None].astype(np.float64)
        raw, state = step_terms(state, ep['p'][t][None], ep['p_next'][t][None], goal[None], v, ep['q_next'][t][None],
                                ep['collided'][t:t + 1], LOWER, UPPER, profile.config)
        rewards.append(profile.combine(raw)[0][0])
        obs_after.append(spec.encode(ep['q_next'][t][None], goal[None], v, state['count'], state['held_once'])[0])
    return np.array(obs_before), np.array(rewards), np.array(obs_after)


def make_buffer(profile, spec, her_ratio, n_step, capacity=H * 8, seed=0):
    return GoalReplay(capacity, H, J, J, np.int8, profile, spec, LOWER, UPPER, her_ratio=her_ratio, n_step=n_step,
                      gamma=0.95, seed=seed)


@pytest.mark.parametrize('n_step', [1, 3])
@pytest.mark.parametrize('her_ratio', [0.0, 1.0])
def test_samples_match_an_independent_live_rollout_for_the_goal_they_carry(profile, spec, her_ratio, n_step):
    rng = np.random.default_rng(1 + n_step)
    episodes = [synthetic_episode(rng) for _ in range(8)]
    buffer = make_buffer(profile, spec, her_ratio, n_step)
    fill(buffer, episodes)
    batch = buffer.sample(120)
    saw_hold = False
    for i in range(120):
        e, t, goal = int(batch['episode'][i]), int(batch['step'][i]), batch['goal'][i].astype(np.float64)
        ep = episodes[e]                                             # slots are assigned in order, one per episode
        obs_before, rewards, obs_after = live_rollout(ep, goal, profile, spec)
        m = min(n_step, H - t)
        expected_return = sum(0.95 ** k * rewards[t + k] for k in range(m))
        np.testing.assert_allclose(batch['obs'][i], obs_before[t], atol=1e-5)
        np.testing.assert_allclose(batch['next_obs'][i], obs_after[t + m - 1], atol=1e-5)
        np.testing.assert_allclose(batch['reward'][i], expected_return, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(batch['discount'][i], 0.95 ** m, rtol=1e-6)
        saw_hold |= bool(obs_after[t + m - 1][-2] >= 1.0)           # count feature saturated: a 5-step dwell occurred
    if her_ratio == 1.0:
        assert saw_hold, 'the relabelled batch never exercised the dwell logic'
        assert batch['relabelled'].all()
    else:
        assert not batch['relabelled'].any()
        np.testing.assert_allclose(batch['goal'], np.array([episodes[int(e)]['goal'] for e in batch['episode']]), atol=1e-6)


def test_relabelled_goal_is_a_position_reached_at_or_after_the_sampled_step(profile, spec):
    rng = np.random.default_rng(5)
    episodes = [synthetic_episode(rng) for _ in range(8)]
    buffer = make_buffer(profile, spec, 1.0, 1)
    fill(buffer, episodes)
    batch = buffer.sample(200)
    for i in range(200):
        ep, t = episodes[int(batch['episode'][i])], int(batch['step'][i])
        reached = ep['p_next'][t:]
        assert np.min(np.linalg.norm(reached - batch['goal'][i], axis=1)) < 1e-6        # goal is returned as float32


def test_n_step_segment_is_truncated_at_the_episode_end_and_uses_gamma_to_the_m(profile, spec):
    rng = np.random.default_rng(6)
    episodes = [synthetic_episode(rng) for _ in range(4)]
    buffer = make_buffer(profile, spec, 0.0, 3)
    fill(buffer, episodes)
    batch = buffer.sample(400)
    last = batch['step'] == H - 1
    assert last.any() and np.allclose(batch['discount'][last], 0.95)
    second_last = batch['step'] == H - 2
    assert second_last.any() and np.allclose(batch['discount'][second_last], 0.95 ** 2)
    early = batch['step'] < H - 3
    assert np.allclose(batch['discount'][early], 0.95 ** 3)


def test_relabel_fraction_follows_her_ratio(profile, spec):
    rng = np.random.default_rng(7)
    buffer = make_buffer(profile, spec, 0.8, 1)
    fill(buffer, [synthetic_episode(rng) for _ in range(8)])
    assert abs(buffer.sample(4000)['relabelled'].mean() - 0.8) < 0.03


def test_only_complete_episodes_are_sampled_and_slots_are_reused_circularly(profile, spec):
    rng = np.random.default_rng(8)
    buffer = make_buffer(profile, spec, 0.0, 1, capacity=H * 4)
    with pytest.raises(RuntimeError, match='no complete episode'):
        buffer.sample(4)
    first = [synthetic_episode(rng) for _ in range(4)]
    fill(buffer, first)
    assert len(buffer) == 4 * H
    buffer.begin(np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))                  # overwrite two slots; not yet complete
    assert len(buffer) == 2 * H
    assert set(buffer.sample(300)['episode'].tolist()) == {2, 3}
