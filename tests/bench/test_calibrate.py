"""Reward calibration: the locked formula, immutability of the locked register, and data provenance."""
import re
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from natural_rl.rewards import RewardNotLocked, load_profile
from natural_rl.rewards.calibrate import AlreadyLocked, lambda_formula, lock_register, table_sha256

from .conftest import ROOT


def test_formula_matches_a_hand_computed_value_and_the_cap_and_floor():
    r0 = np.array([-10.0, 20.0, -30.0, 40.0])                              # median |r0| = 25 -> scale 2.5
    sums = {'a': np.array([0.5, 1.0, 1.5, 2.0, 2.5]), 'tiny': np.zeros(5), 'big': np.array([100.0] * 5)}
    got = lambda_formula(r0, sums)
    assert np.isclose(got['a'], 2.5 / 2.0)                                  # P75 of |a| = 2.0
    assert got['tiny'] == 10.0                                              # floor 1e-4 would give 25000: capped at 10
    assert np.isclose(got['big'], 2.5 / 100.0)


def test_formula_uses_absolute_values_and_rejects_empty_pilot_data():
    assert np.isclose(lambda_formula([-4.0] * 6, {'x': -np.ones(6) * 0.2})['x'], 0.4 / 0.2)
    with pytest.raises(ValueError, match='no pilot'):
        lambda_formula([], {'x': [1.0]})


def test_lock_writes_coefficients_and_provenance_then_refuses_a_second_lock(tmp_path):
    register = tmp_path / 'reward_v1.yaml'
    shutil.copy(ROOT + '/configs/rewards/reward_v1.yaml', register)
    text = re.sub(r'^status: \w+', 'status: unlocked', register.read_text(), flags=re.M)             # the shipped register may be locked
    register.write_text(re.sub(r'^    (time|path|speed|accel|rom): .*$', r'    \1: null', text, flags=re.M))
    lambdas = {'time': 1.5, 'path': 2.25, 'speed': 0.125, 'accel': 3.0, 'rom': 10.0}
    lock_register(register, lambdas, 1234, 'ab' * 32, 'cd' * 32, 'unit-test pilot')
    cfg = yaml.safe_load(register.read_text())
    assert cfg['status'] == 'locked' and cfg['calibration']['data']['n_episodes'] == 1234
    assert cfg['calibration']['data']['sha256'] == 'ab' * 32 and cfg['calibration']['lambda'] == lambdas
    assert '# Reward register v1' in register.read_text()                   # comments survive the rewrite
    assert isinstance(load_profile(cfg, 'r7_balanced'), object)             # now every profile builds
    with pytest.raises(AlreadyLocked):
        lock_register(register, lambdas, 1, 'x', 'y', 'z')


def test_the_shipped_register_is_either_unlocked_and_empty_or_locked_with_complete_provenance():
    cfg = yaml.safe_load(open(ROOT + '/configs/rewards/reward_v1.yaml'))
    lam = cfg['calibration']['lambda']
    if cfg['status'] == 'unlocked':
        assert all(v is None for v in lam.values())
        with pytest.raises(RewardNotLocked):
            load_profile(cfg, 'r2_time')
    else:
        data = cfg['calibration']['data']
        assert cfg['status'] == 'locked' and all(v is not None and v > 0 for v in lam.values())
        assert data['sha256'] and data['code_sha'] and data['n_episodes'] and data['locked_at']
        assert load_profile(cfg, 'r7_balanced') is not None


def test_data_hash_changes_when_any_pilot_value_changes():
    a = pd.DataFrame({'sum_r0': [1.0, 2.0], 'sum_path': [0.1, 0.2]})
    b = a.copy()
    b.loc[1, 'sum_path'] = 0.2000001
    assert table_sha256(a) == table_sha256(a.copy()) and table_sha256(a) != table_sha256(b)
