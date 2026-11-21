"""Shared analytic reference trajectories for the natural_rl metric tests (no simulator needed)."""
import os

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope='session')
def metrics_cfg():
    from natural_rl.metrics.kinematics import KinematicsConfig
    with open(os.path.join(ROOT, 'configs', 'metrics', 'metrics_v1.yaml')) as handle:
        return KinematicsConfig.from_yaml(yaml.safe_load(handle))


def min_jerk_position(tau):
    """Normalised minimum-jerk position profile 10 t^3 - 15 t^4 + 6 t^5."""
    tau = np.asarray(tau, dtype=np.float64)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def min_jerk_reach(start, goal, duration, dt=0.05, lead=4, hold=12, joint_delta=None):
    """Executed samples of a straight minimum-jerk reach: `lead` still samples, the movement, `hold` still samples.

    Returns (ee_xyz (N, 3), q (N, 7)). q moves monotonically along joint_delta (default: a fixed 7-vector).
    """
    start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    n_move = int(round(duration / dt))
    x = min_jerk_position(np.linspace(0.0, 1.0, n_move + 1))
    profile = np.concatenate([np.zeros(lead), x, np.ones(hold)])
    ee = start + profile[:, None] * (goal - start)
    delta = np.linspace(0.2, 0.8, 7) if joint_delta is None else np.asarray(joint_delta, dtype=np.float64)
    return ee, 0.1 + profile[:, None] * delta


def concatenate_reaches(waypoints, durations, pause_samples, dt=0.05, lead=4, hold=12):
    """A multi-sub-movement path through waypoints, with `pause_samples` of rest between sub-movements."""
    pieces_ee, pieces_q = [], []
    for i, duration in enumerate(durations):
        ee, q = min_jerk_reach(waypoints[i], waypoints[i + 1], duration, dt, lead=lead if i == 0 else 0, hold=0)
        pieces_ee.append(ee)
        pieces_q.append(q)
        if i < len(durations) - 1:
            pieces_ee.append(np.repeat(ee[-1:], pause_samples, axis=0))
            pieces_q.append(np.repeat(q[-1:], pause_samples, axis=0))
    pieces_ee.append(np.repeat(pieces_ee[-1][-1:], hold, axis=0))
    pieces_q.append(np.repeat(pieces_q[-1][-1:], hold, axis=0))
    return np.concatenate(pieces_ee), np.concatenate(pieces_q)
