"""P3 selection reads validation tables only, ranks by terminal hold then unsafe rate, and refuses incomplete data."""
import os

import numpy as np
import pandas as pd
import pytest

from natural_rl.checkpoints import step_name
from natural_rl.models.registry import MODEL_IDS
from natural_rl.programme import run_id
from natural_rl.selection import p3_selection


def fake_run(root, model, seed, val_hold, val_unsafe, test_hold=0.0):
    budget = 'budget_moderate_v1' if model == 'sac_pidlag' else None
    rid = run_id(model, 'r7_balanced', seed, budget=budget)
    run = root / 'runs' / 'kinematic' / model / 'r7_balanced' / rid
    (run / 'eval' / 'kinematic' / 'val_v1' / step_name(1000)).mkdir(parents=True)
    (run / 'eval' / 'kinematic' / 'test_v1' / step_name(1000)).mkdir(parents=True)
    (run / 'manifest.json').write_text('{}')
    pd.DataFrame({'checkpoint_step': [1000], 'path': ['x'], 'sha256': ['y'], 'saved_at': ['z'], 'kind': ['final'],
                  'video_status': ['ok']}).to_csv(run / 'checkpoint_index.csv', index=False)
    n = 100
    val = pd.DataFrame({'terminal_hold': np.arange(n) < int(val_hold * n), 'unsafe_events': np.full(n, val_unsafe * 100)})
    test = pd.DataFrame({'terminal_hold': np.arange(n) < int(test_hold * n), 'unsafe_events': np.zeros(n)})
    val.to_parquet(run / 'eval' / 'kinematic' / 'val_v1' / step_name(1000) / 'episode_metrics.parquet')
    test.to_parquet(run / 'eval' / 'kinematic' / 'test_v1' / step_name(1000) / 'episode_metrics.parquet')


@pytest.fixture
def root(tmp_path):
    os.environ['NATURAL_RL_RESULTS'] = str(tmp_path)
    yield tmp_path
    os.environ.pop('NATURAL_RL_RESULTS', None)


def populate(root, holds, unsafe=None, test_holds=None):
    for k, model in enumerate(m for m in MODEL_IDS if m != 'decqn'):
        for seed in range(5):
            fake_run(root, model, seed, holds[k], (unsafe or [0.01] * 9)[k], (test_holds or [0.0] * 9)[k])


def test_selects_decqn_plus_the_three_highest_validation_hold_rates(root):
    models = [m for m in MODEL_IDS if m != 'decqn']
    holds = [0.10, 0.50, 0.20, 0.90, 0.30, 0.60, 0.40, 0.70, 0.80]
    populate(root, holds)
    chosen, table = p3_selection()
    ranked = [models[i] for i in np.argsort(holds)[::-1][:3]]
    assert chosen == ['decqn'] + ranked and len(table) == 9


def test_ties_are_broken_by_the_lower_unsafe_rate(root):
    models = [m for m in MODEL_IDS if m != 'decqn']
    holds = [0.5] * 9
    unsafe = [0.09, 0.01, 0.05, 0.02, 0.08, 0.03, 0.07, 0.06, 0.04]
    populate(root, holds, unsafe)
    chosen, _ = p3_selection()
    assert chosen[1:] == [models[1], models[3], models[5]]                         # the three lowest unsafe rates


def test_test_results_cannot_influence_the_choice(root):
    models = [m for m in MODEL_IDS if m != 'decqn']
    holds = [0.1, 0.5, 0.2, 0.9, 0.3, 0.6, 0.4, 0.7, 0.8]
    populate(root, holds, test_holds=[1.0, 0, 0, 0, 0, 0, 0, 0, 0])               # the worst validation model is perfect on test
    assert models[0] not in p3_selection()[0]


def test_incomplete_p2_data_is_refused_with_the_missing_runs_named(root):
    populate(root, [0.5] * 9)
    import shutil
    shutil.rmtree(next((root / 'runs' / 'kinematic' / 'sac' / 'r7_balanced').glob('*-s03')))
    with pytest.raises(FileNotFoundError, match='sac-r7-s03'):
        p3_selection()
