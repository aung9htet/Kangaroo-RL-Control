"""Every implemented controller trains on the real scene, checkpoints, reloads and acts under the common action contract.

Slow (each run builds the scene). No videos here (the end-to-end test covers those); this checks the model adapters:
vectorised environment, goal-relabelling buffer, checkpoint callback, policy reload and the model-specific logs.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from natural_rl.cli import main
from natural_rl.models.registry import MODEL_IDS, implemented, load_policy

pytestmark = [pytest.mark.sim, pytest.mark.slow]

OFF = ['hyper.learning_starts=1600', 'hyper.replay=16000']
SETTINGS = {
    'ppo_md': ['hyper.n_steps=100', 'hyper.batch_size=160'], 'ppo_box': ['hyper.n_steps=100', 'hyper.batch_size=160'],
    'sac': OFF, 'td3': OFF, 'tqc': OFF, 'crossq': OFF, 'sac_pidlag': OFF,
    'tdmpc2': ['hyper.seed_steps=200', 'hyper.replay=5000', 'hyper.iterations=2', 'hyper.num_samples=64', 'hyper.num_elites=8',
               'hyper.num_pi_trajs=8', 'hyper.batch_size=32'],
    'decqn': ['hyper.warmup=800', 'hyper.replay=16000'], 'growing_q': ['hyper.warmup=800', 'hyper.replay=16000'],
}
KIND = {'decqn': 'multidiscrete', 'growing_q': 'multidiscrete', 'ppo_md': 'multidiscrete'}


@pytest.fixture(scope='module')
def results_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('controllers')
    os.environ['NATURAL_RL_RESULTS'] = str(root)
    yield root
    os.environ.pop('NATURAL_RL_RESULTS', None)


def test_every_registered_model_has_a_config_and_the_implemented_set_is_what_the_runbook_says():
    from natural_rl.registry import model_config
    for model in MODEL_IDS:
        assert model_config(model)['id'] == model
    assert set(implemented()) == set(MODEL_IDS)


def test_the_two_ppo_configurations_differ_only_in_the_action_distribution():
    from natural_rl.registry import model_config
    assert model_config('ppo_md')['hyper'] == model_config('ppo_box')['hyper']
    assert model_config('ppo_md')['n_envs'] == model_config('ppo_box')['n_envs']


@pytest.mark.parametrize('model', list(MODEL_IDS))
def test_controller_trains_checkpoints_reloads_and_obeys_the_action_contract(model, results_root):
    run_id = f'ctl-{model}'
    argv = ['train', '--model', model, '--reward', 'r1_hold', '--suite', 'train_v1', '--steps', '2400', '--seed', '0',
            '--run-id', run_id, '--checkpoint-every', '1200', '--smoke', '--set', *SETTINGS[model]]
    if model == 'sac_pidlag':
        argv += ['--cost-budget-id', 'budget_moderate_v1']
    assert main(argv) == 0
    run = next((results_root / 'smoke' / 'runs').glob(f'*/{model}/*/{run_id}'))
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['n_checkpoints'] == 3                # steps 0, 1200, 2400 exactly
    index = pd.read_csv(run / 'checkpoint_index.csv')
    assert list(index['checkpoint_step']) == [0, 1200, 2400]
    episodes = pd.read_parquet(run / 'train_episodes.parquet')
    assert len(episodes) >= 16 and {'episode_return', 'episodic_cost', 'final_distance_m'} <= set(episodes.columns)
    assert any(c.startswith('weighted_sum_') for c in episodes.columns)

    spec = _spec()
    obs = np.random.default_rng(0).normal(size=(5, spec.dim)).astype(np.float32)
    for step in (0, 2400):
        policy = load_policy(model, run / 'checkpoints' / f'step_{step:09d}' / 'policy_state', spec)
        def replay():                                   # planners (TD-MPC2) are deterministic given reset + seed
            policy.reset()
            policy.seed(7)
            return policy.act(obs, deterministic=True)
        action = replay()
        assert action.shape == (5, 7)
        np.testing.assert_array_equal(action, replay())                                        # no exploration noise at replay
        if KIND.get(model, 'box') == 'multidiscrete':
            assert np.issubdtype(action.dtype, np.integer) and action.min() >= 0 and action.max() <= 6
        else:
            assert action.min() >= -1.0 - 1e-6 and action.max() <= 1.0 + 1e-6
    if model == 'sac_pidlag':
        history = pd.read_parquet(run / 'train_history.parquet')
        assert {'lambda_pid', 'cost_ema', 'budget'} <= set(history.columns) and (history['lambda_pid'].dropna() >= 0).all()
        assert (history['budget'].dropna() == 0.5).all()
    if model == 'growing_q':
        first = load_policy(model, run / 'checkpoints' / 'step_000000000' / 'policy_state', spec)
        assert first.info()['active_bins'] == 2                                                # coarsest sub-space at step 0
        assert load_policy(model, run / 'checkpoints' / 'step_000002400' / 'policy_state', spec).info()['active_bins'] == 7


def _spec():
    from kangaroo_rl.config import load_config
    from kangaroo_rl.experiment import make_experiment
    from natural_rl.env.contract import ObsSpec
    from natural_rl.registry import repo_root
    exp = make_experiment(load_config(repo_root() / 'configs' / 'reach.yaml'))
    return ObsSpec('v2', exp.scene.lower, exp.scene.upper, exp.pool.lower, exp.pool.upper)


def test_a_registered_but_unwritten_model_fails_loudly_and_is_never_substituted(results_root, monkeypatch):
    from natural_rl.models import registry
    monkeypatch.setattr(registry, '_IMPLEMENTATIONS', {k: v for k, v in registry._IMPLEMENTATIONS.items() if k != 'td3'})
    assert main(['train', '--model', 'td3', '--reward', 'r1_hold', '--suite', 'train_v1', '--steps', '1600', '--seed', '0',
                 '--run-id', 'ctl-td3x', '--checkpoint-every', '800', '--smoke']) == 2


def test_a_calibrated_reward_profile_refuses_to_train_while_the_register_is_unlocked(results_root, capsys):
    assert main(['train', '--model', 'decqn', '--reward', 'r7_balanced', '--suite', 'train_v1', '--steps', '1600', '--seed', '0',
                 '--run-id', 'ctl-locked', '--checkpoint-every', '800', '--smoke']) == 2
    assert 'calibrate' in capsys.readouterr().err


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='no GPU in this container (use compose.gpu.yaml)')
@pytest.mark.parametrize('model', list(MODEL_IDS))
def test_every_controller_also_trains_on_the_gpu_and_reloads(model, results_root):
    run_id = f'gpu-{model}'
    argv = ['train', '--model', model, '--reward', 'r1_hold', '--suite', 'train_v1', '--steps', '2400', '--seed', '0',
            '--run-id', run_id, '--checkpoint-every', '1200', '--smoke', '--device', 'cuda', '--set', *SETTINGS[model]]
    if model == 'sac_pidlag':
        argv += ['--cost-budget-id', 'budget_moderate_v1']
    assert main(argv) == 0
    run = next((results_root / 'smoke' / 'runs').glob(f'*/{model}/*/{run_id}'))
    assert json.loads((run / 'manifest.json').read_text())['status'] == 'complete'
    policy = load_policy(model, run / 'checkpoints' / 'step_000002400' / 'policy_state', _spec())
    obs = np.random.default_rng(0).normal(size=(3, _spec().dim)).astype(np.float32)
    assert policy.act(obs).shape == (3, 7)
