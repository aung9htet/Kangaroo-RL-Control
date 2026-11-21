"""Accuracy measures shared by the model and simulator evaluations."""
from __future__ import annotations

import numpy as np

THRESHOLDS = (0.01, 0.02, 0.05)   # m
HOLD_WINDOW = 1.0                 # s: the last second of an episode


def summarise(distances, dt, collided=None):
    """distances: (n_goals, H) reach-point error after every step. Returns a flat dict of measures."""
    distances = np.asarray(distances)
    final = distances[:, -1]
    within = distances <= 0.02
    reached = within.any(axis=1)
    first = np.where(reached, within.argmax(axis=1) + 1, -1)
    window = max(1, int(round(HOLD_WINDOW / dt)))
    result = {
        'final_median': float(np.median(final)),
        'final_p90': float(np.percentile(final, 90)),
        'final_max': float(final.max()),
        'final_mean': float(final.mean()),
        'reached_2cm': float(reached.mean()),
        'time_to_2cm_median': float(np.median(first[reached]) * dt) if reached.any() else float('nan'),
        'hold_2cm': float(within[:, -window:].mean()),
        'n_goals': int(len(final)),
    }
    for threshold in THRESHOLDS:
        result[f'success_{int(round(threshold * 100))}cm'] = float((final <= threshold).mean())
    if collided is not None:
        result['rejected'] = float(np.mean(collided))
    return result
