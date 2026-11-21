"""Task outcomes computed from executed (accepted) end-effector positions, never from commands.

Timing convention (reported with every result): sample k is the accepted state after k decisions, at time
k * dt, with k = 0 the initial state. A 5-step hold therefore spans 5 samples = 0.20 s between its first and
last sample. Success is always computed independently of the reward that was used for training.
"""
from __future__ import annotations

import numpy as np


def distances(ee_xyz, goal_xyz) -> np.ndarray:
    """d_k = ||p_k - g|| for every sample k (metres)."""
    return np.linalg.norm(np.asarray(ee_xyz, dtype=np.float64) - np.asarray(goal_xyz, dtype=np.float64), axis=-1)


def terminal_run_start(d: np.ndarray, tolerance: float) -> int | None:
    """Smallest k such that d_j <= tolerance for every j >= k (None if the last sample is outside)."""
    inside = d <= tolerance
    if len(d) == 0 or not inside[-1]:
        return None
    outside = np.flatnonzero(~inside)
    return 0 if len(outside) == 0 else int(outside[-1] + 1)


def episode_outcomes(ee_xyz, goal_xyz, dt: float, tolerance: float = 0.02, hold_steps: int = 5,
                     terminal_window: int = 5) -> dict:
    """First entry, any hold, terminal hold, final error, departures and path length of one episode.

    ee_xyz has N = decisions + 1 samples (k = 0 is the initial state). Episodes that end early are scored on
    the samples they have; nothing is padded.
    """
    p = np.asarray(ee_xyz, dtype=np.float64)
    d = distances(p, goal_xyz)
    n = len(d)
    inside = d <= tolerance
    # First entry is an accepted position after at least one decision (k >= 1).
    entered = np.flatnonzero(inside[1:]) + 1 if n > 1 else np.array([], dtype=int)
    first_entry_k = int(entered[0]) if len(entered) else None

    run = 0
    any_hold = False
    for flag in inside:
        run = run + 1 if flag else 0
        if run >= hold_steps:
            any_hold = True
            break

    terminal_hold = bool(n >= terminal_window and inside[-terminal_window:].all())
    departures = 0
    if first_entry_k is not None:
        after = inside[first_entry_k:]
        departures = int(np.sum(after[:-1] & ~after[1:]))   # inside -> outside transitions after first entry
    path = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if n > 1 else 0.0
    return {
        'first_entry_s': float('nan') if first_entry_k is None else first_entry_k * dt,
        'first_entry': first_entry_k is not None,
        'any_hold': bool(any_hold),
        'terminal_hold': terminal_hold,
        'final_error_m': float(d[-1]),
        'path_m': path,
        'duration_s': float((n - 1) * dt),
        'n_decisions': int(n - 1),
        'goal_departures': departures,
    }
