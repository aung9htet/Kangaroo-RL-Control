"""P0: the historical DecQN loop runs unchanged under the benchmark checkpoint/video contract (slow; real scene)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from natural_rl.cli import main
from natural_rl.hist import parity_differences

pytestmark = [pytest.mark.sim, pytest.mark.slow]


def test_parity_differences_reports_exactly_the_keys_that_differ():
    a = {'train': {'steps': 1, 'lr': 2}, 'agent': {'x': 1}}
    b = {'train': {'steps': 5, 'lr': 2}, 'agent': {'x': 1}, 'extra': 0}
    assert parity_differences(a, b) == ['extra', 'train.steps']


def test_hist_contract_requires_decqn_with_the_original_reward(tmp_path, capsys):
    os.environ['NATURAL_RL_RESULTS'] = str(tmp_path)
    try:
        for model, reward in (('sac', 'r0_original'), ('decqn', 'r1_hold')):                 # refused cleanly (exit 2) with the reason
            assert main(['train', '--contract', 'hist_v1', '--model', model, '--reward', reward, '--steps', '1600', '--run-id', 'x',
                         '--checkpoint-every', '800', '--smoke']) == 2
            assert 'hist_v1' in capsys.readouterr().err
    finally:
        os.environ.pop('NATURAL_RL_RESULTS', None)


def test_historical_loop_runs_with_checkpoint_videos_and_the_converted_policy_equals_the_native_agent(tmp_path):
    os.environ['NATURAL_RL_RESULTS'] = str(tmp_path)
    try:
        assert main(['train', '--contract', 'hist_v1', '--model', 'decqn', '--reward', 'r0_original', '--suite', 'train_v1',
                     '--steps', '1600', '--seed', '0', '--run-id', 'decqn-r0hist-s00', '--checkpoint-every', '800',
                     '--record-every-checkpoint', '--smoke', '--set', 'train.warmup=800', 'train.replay=16000',
                     'train.eval_goals=32', 'train.eval_every=1600']) == 0
        run = next((tmp_path / 'smoke' / 'runs').glob('*/decqn/r0_original/decqn-r0hist-s00'))
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['status'] == 'complete' and manifest['contract'] == 'hist_v1' and manifest['obs_version'] == 'v1'
        assert set(manifest['parity_config_differences']) == {'train.warmup', 'train.replay', 'train.eval_goals', 'train.eval_every'}
        assert not manifest['parity_with_reach_yaml']                         # a --set run is never a parity run
        assert list(pd.read_csv(run / 'checkpoint_index.csv')['checkpoint_step']) == [0, 800, 1600]
        assert main(['video', 'verify', '--run-id', 'decqn-r0hist-s00', '--all-checkpoints']) == 0
        for name in ('config.yaml', 'run.json', 'metrics.csv'):       # train.csv is only written every 1000 updates (historical behaviour)
            assert (run / 'native' / name).exists(), name
        assert (run / 'native' / 'checkpoints' / 'step_00001600.pt').exists()

        # the replay-parity claim: native historical agent and converted benchmark policy choose the same actions
        from kangaroo_rl.evaluate import load_agent
        from natural_rl.env.contract import ObsSpec
        from natural_rl.models.registry import load_policy
        agent, state = load_agent(run / 'native' / 'checkpoints' / 'step_00001600.pt')
        cfg = state['network_config']
        spec = ObsSpec('v1', np.array(cfg['q_lower']), np.array(cfg['q_upper']), np.array(cfg['goal_lower']), np.array(cfg['goal_upper']))
        policy = load_policy('decqn', run / 'checkpoints' / 'step_000001600' / 'policy_state', spec)
        rng = np.random.default_rng(0)
        q = rng.uniform(spec.q_lower, spec.q_upper, (200, 7))
        g = rng.uniform(spec.goal_lower, spec.goal_upper, (200, 3))
        np.testing.assert_array_equal(agent.act(q, g), policy.act(spec.encode(q, g)))
    finally:
        os.environ.pop('NATURAL_RL_RESULTS', None)
