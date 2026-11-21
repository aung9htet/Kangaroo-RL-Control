"""Fast tests need only numpy/torch; tests marked `sim` build the LIBERO scene (run them in the container)."""
import os

import numpy as np
import pytest

from kangaroo_rl.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    config.addinivalue_line('markers', 'sim: needs Kangaroo_Libero (LIBERO + MuJoCo scene)')


@pytest.fixture(scope='session')
def reach_config():
    return load_config(os.path.join(ROOT, 'configs', 'reach.yaml'))


@pytest.fixture(scope='session')
def experiment(reach_config):
    pytest.importorskip('kangaroo_libero')
    from kangaroo_rl.config import override
    from kangaroo_rl.experiment import make_experiment
    return make_experiment(override(reach_config, ['goals.fk_samples=200000']), keep_env=True)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
