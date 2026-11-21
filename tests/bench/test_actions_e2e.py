"""Training, checkpoints, videos and evaluation under every action mode and state, on the real scene (slow)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from natural_rl.cli import main
from natural_rl.models.registry import load_policy

from .test_controllers import SETTINGS

pytestmark = [pytest.mark.sim, pytest.mark.slow]


@pytest.fixture(scope='module')
def results_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('actions')
    os.environ['NATURAL_RL_RESULTS'] = str(root)
    yield root
    os.environ.pop('NATURAL_RL_RESULTS', None)


def train(model, run_id, action, obs=None, video=False, steps=1600):
    args = ['train', '--model', model, '--reward', 'r1_hold', '--suite', 'train_v1', '--steps', str(steps), '--seed', '0',
            '--run-id', run_id, '--checkpoint-every', str(steps), '--smoke', '--action', action, '--set', *SETTINGS[model]]
    if obs:
        args[args.index('--smoke'):args.index('--smoke')] = ['--obs', obs]
    if video:
        args.insert(args.index('--smoke'), '--record-every-checkpoint')
    assert main(args) == 0
    return args


def run_path(root, run_id):
    return next((root / 'smoke' / 'runs').glob(f'*/*/*/{run_id}'))


def test_a_torque_run_trains_on_the_dynamics_backend_with_verified_videos_and_is_evaluated_there(results_root, capsys):
    train('decqn', 'ae-decqn-tor', 'torque', video=True)
    out = run_path(results_root, 'ae-decqn-tor')
    assert out.parts[-4] == 'dynamics'
    manifest = json.loads((out / 'manifest.json').read_text())
    assert (manifest['action_mode'], manifest['obs_version'], manifest['backend'], manifest['status']) == ('torque', 'v3', 'dynamics', 'complete')
    assert 'dynamics' in manifest['config_hashes']
    progress = json.loads((out / 'progress.json').read_text())                          # what run-all's overall display reads
    assert progress['step'] == progress['total'] == 1600 and progress['cell'] == 'torque/v3'
    index = pd.read_csv(out / 'video_index.csv')
    assert len(index) == 2 and (index['status'] == 'ok').all()                         # step 0 and the final checkpoint, both verified
    assert main(['video', 'verify', '--run-id', 'ae-decqn-tor', '--all-checkpoints']) == 0
    assert main(['eval', '--run-id', 'ae-decqn-tor', '--suite', 'val_v1', '--final']) == 0       # default backend = the run's own
    tables = list((out / 'eval' / 'dynamics' / 'val_v1').glob('step_*/episode_metrics.parquet'))
    assert len(tables) == 1 and not (out / 'eval' / 'kinematic').exists()
    metrics = pd.read_parquet(tables[0])
    assert (metrics['backend'] == 'dynamics').all() and len(metrics) == 100
    assert main(['analyse', '--run-id', 'ae-decqn-tor']) == 0
    assert main(['eval', '--run-id', 'ae-decqn-tor', '--suite', 'val_v1', '--final', '--backend', 'physics']) == 2             # refused cleanly, not a crash


def test_torque_cannot_be_evaluated_on_the_full_scene_physics_backend(results_root):
    from natural_rl.cli import _run_context
    from natural_rl.eval import evaluate_checkpoint
    train('decqn', 'ae-decqn-tor2', 'torque')
    out, manifest, scene, spec, profile = _run_context('ae-decqn-tor2', keep_env=True)
    with pytest.raises(NotImplementedError, match='position targets'):
        evaluate_checkpoint(out, manifest, scene, spec, profile, 1600, 'val_v1', backend='physics', max_tasks=1)


@pytest.mark.parametrize('model,action,obs', [('sac', 'torque', None), ('ppo_md', 'position', 'v3'), ('ppo_box', 'velocity', 'v1'),
                                               ('td3', 'position', 'v1'), ('tdmpc2', 'torque', 'v2'), ('sac_pidlag', 'torque', 'v3'),
                                               ('decqn', 'velocity', 'v8'), ('sac', 'torque', 'v7'), ('ppo_box', 'velocity', 'q,goal,ee,ee_vel'),
                                               ('tdmpc2', 'position', 'v5')])
def test_every_controller_family_trains_and_reloads_under_other_actions_and_states(results_root, model, action, obs):
    run_id = f'ae-{model}-{action[:3]}-{abs(hash(obs)) % 10000}'
    args = ['train', '--model', model, '--reward', 'r1_hold', '--suite', 'train_v1', '--steps', '1600', '--seed', '0', '--run-id', run_id,
            '--checkpoint-every', '1600', '--smoke', '--action', action, '--set', *SETTINGS[model]]
    if obs:
        args[args.index('--smoke'):args.index('--smoke')] = ['--obs', obs]
    if model == 'sac_pidlag':
        args[args.index('--smoke'):args.index('--smoke')] = ['--cost-budget-id', 'budget_moderate_v1']
    assert main(args) == 0
    out = run_path(results_root, run_id)
    manifest = json.loads((out / 'manifest.json').read_text())
    from natural_rl.cli import _run_context
    _, _, _, spec, _ = _run_context(run_id)
    assert spec.action_mode == action and manifest['status'] == 'complete'
    assert spec.version == (obs or ('v3' if action == 'torque' else 'v2'))
    policy = load_policy(model, out / 'checkpoints' / 'step_000001600' / 'policy_state', spec)
    action_out = policy.act(np.zeros((3, spec.dim), dtype=np.float32), deterministic=True)
    assert np.asarray(action_out).shape == (3, 7)


def test_position_runs_can_be_evaluated_on_the_physics_stratum_without_velocity_feedforward(results_root):
    from natural_rl.cli import _run_context
    from natural_rl.eval import evaluate_checkpoint
    train('decqn', 'ae-decqn-pos', 'position')
    out, manifest, scene, spec, profile = _run_context('ae-decqn-pos', keep_env=True)
    metrics = evaluate_checkpoint(out, manifest, scene, spec, profile, 1600, 'val_v1', backend='physics', max_tasks=2)
    assert len(metrics) == 2 and (metrics['backend'] == 'physics').all()
