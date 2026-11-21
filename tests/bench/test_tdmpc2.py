"""TD-MPC2 adapter: official agent plans, windows equal live rollouts, per-environment planner state, deterministic replay."""
import numpy as np
import pytest
import torch
import yaml

from natural_rl.models.tdmpc2 import SequenceBuffer, TDMPC2Policy, build_cfg
from natural_rl.replay import GoalReplay
from natural_rl.vendor.tdmpc2 import TDMPC2

from .conftest import ROOT
from .test_replay import H, J, LOWER, UPPER, fill, live_rollout, profile, spec, synthetic_episode  # noqa: F401 (fixtures)

HYPER = yaml.safe_load(open(f'{ROOT}/configs/models/tdmpc2.yaml'))['hyper']
FAST = {**HYPER, 'iterations': 2, 'num_samples': 64, 'num_elites': 8, 'num_pi_trajs': 8, 'batch_size': 32}


def make_agent(device='cpu', hyper=FAST, obs_dim=99):
    return TDMPC2(build_cfg(hyper, obs_dim, 7, 100, device))


def test_official_agent_plans_inside_the_actuator_range_with_the_task_discount():
    agent = make_agent()
    assert abs(float(agent.discount) - 0.95) < 1e-9
    a = agent.act(torch.zeros(99), t0=True, eval_mode=True)
    assert tuple(a.shape) == (7,) and bool(torch.isfinite(a).all()) and float(a.abs().max()) <= 1.0


def sequence_buffer(profile, spec, her):
    replay = GoalReplay(H * 8, H, J, J, np.float32, profile, spec, LOWER, UPPER, her_ratio=her, n_step=1, gamma=0.95, seed=3)
    rng = np.random.default_rng(4)
    episodes = [synthetic_episode(rng) for _ in range(8)]
    for ep in episodes:
        ep['action'] = ep['action'].astype(np.float32) / 3.0 - 1.0
    fill(replay, episodes)
    return SequenceBuffer(replay, 3, 64, 'cpu'), episodes


@pytest.mark.parametrize('her', [0.0, 1.0])
def test_windows_equal_an_independent_live_rollout_for_their_own_goal(profile, spec, her):
    buf, episodes = sequence_buffer(profile, spec, her)
    a = buf.sample_arrays()
    for i in range(64):
        e, s, goal = int(a['episode'][i]), int(a['start'][i]), a['goal'][i]
        obs_before, rewards, _ = live_rollout(episodes[e], goal, profile, spec)
        np.testing.assert_allclose(a['obs'][:, i], obs_before[s:s + 4], atol=1e-5)
        np.testing.assert_allclose(a['reward'][:, i, 0], rewards[s:s + 3], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(a['action'][:, i], episodes[e]['action'][s:s + 3], atol=1e-6)
    assert a['obs'].shape == (4, 64, spec.dim) and a['action'].shape == (3, 64, 7) and a['reward'].shape == (3, 64, 1)
    assert a['start'].max() <= H - 4                                          # the window never leaves the episode


def test_agent_update_runs_on_relabelled_windows_and_lowers_the_loss_on_a_fixed_batch(profile, spec):
    buf, _ = sequence_buffer(profile, spec, 0.8)
    agent = make_agent(obs_dim=spec.dim)
    buf.batch = int(agent.cfg.batch_size)                                     # as in training: one batch size from the config
    torch.manual_seed(0)
    fixed = buf.sample()
    buf.sample = lambda: fixed
    first = float(agent.update(buf)['total_loss'])
    for _ in range(60):
        last = float(agent.update(buf)['total_loss'])
    obs, action, reward, terminated, task = fixed
    assert obs.shape[0] == 4 and action.shape[0] == 3 and float(terminated.abs().sum()) == 0 and task is None
    assert last < first and np.isfinite(last)


def test_policy_keeps_a_planner_state_per_environment_and_is_deterministic_for_a_seed(tmp_path):
    agent = make_agent()
    policy = TDMPC2Policy(agent, agent.cfg)
    obs = np.random.default_rng(0).normal(size=(3, 99)).astype(np.float32)

    def run():
        policy.reset()
        policy.seed(11)
        return [policy.act(obs), policy.act(obs * 0.5)]

    a, b = run(), run()
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[0].shape == (3, 7) and np.abs(a[0]).max() <= 1.0
    assert policy.prev.shape == (3, 3, 7) and not torch.equal(policy.prev[0], policy.prev[1])   # states are not shared


def test_saved_policy_reloads_to_identical_actions_on_the_same_device(tmp_path, monkeypatch):
    monkeypatch.setenv('NATURAL_RL_POLICY_DEVICE', 'cpu')
    agent = make_agent()
    policy = TDMPC2Policy(agent, agent.cfg)
    policy.save(tmp_path)
    reloaded = TDMPC2Policy.load(tmp_path, None)
    obs = np.random.default_rng(1).normal(size=(2, 99)).astype(np.float32)
    def run(p):
        p.reset()
        p.seed(5)                                                              # the RNG is global: seed right before each use
        return [p.act(obs), p.act(obs)]
    for a, b in zip(run(policy), run(reloaded)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU in this container (use compose.gpu.yaml)')
def test_agent_plans_and_updates_on_the_gpu(profile, spec):
    agent = make_agent('cuda', obs_dim=spec.dim)
    a = agent.act(torch.zeros(spec.dim), t0=True, eval_mode=True)
    assert a.device.type == 'cpu' and bool(torch.isfinite(a).all())
    buf, _ = sequence_buffer(profile, spec, 0.8)
    buf.batch, buf.device = int(agent.cfg.batch_size), torch.device('cuda')
    assert np.isfinite(float(agent.update(buf)['total_loss']))
