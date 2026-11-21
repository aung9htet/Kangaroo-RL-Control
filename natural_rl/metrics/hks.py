"""Human-like Kinematic Proxy Score (HKS) and reliability-adjusted R-HKS.

HKS (0-100) averages four equally weighted interpretable domains, each built from frozen 5-95 % scalers
fitted on successful, quality-valid *validation* episodes only (never test data). It measures resemblance to a
specific set of simple reaching-kinematic priors; it is not a validated measure of perceived naturalness.

    D1_smooth     = 0.5 b_up(SPARC)            + 0.5 b_down(log1p DJ)
    D2_timing     = 0.5 b_down(template RMSE)  + 0.5 b_down(max(peaks - 1, 0))
    D3_spatial    = b_up(path efficiency)                        (registered direct-reach tasks only)
    D4_continuity = 0.5 b_down(pause fraction) + 0.5 b_down(joint reversal rate)
    HKS           = 25 (D1 + D2 + D3 + D4)
    R_HKS_i       = terminal_hold_i * HKS_i   (failed, unscorable or non-finite episodes contribute zero)
"""
from __future__ import annotations

import math

import numpy as np

# metric -> direction; 'up' means larger is more human-like, 'down' means smaller is.
SCALED_METRICS = {
    'sparc': 'up',
    'log1p_dj': 'down',
    'template_rmse': 'down',
    'peaks_excess': 'down',
    'path_efficiency': 'up',
    'pause_fraction': 'down',
    'joint_reversal_rate': 'down',
}
DOMAINS = {
    'D1_smooth': (('sparc', 0.5), ('log1p_dj', 0.5)),
    'D2_timing': (('template_rmse', 0.5), ('peaks_excess', 0.5)),
    'D3_spatial': (('path_efficiency', 1.0),),
    'D4_continuity': (('pause_fraction', 0.5), ('joint_reversal_rate', 0.5)),
}


class ScalerError(RuntimeError):
    """Blocking metric-calibration error: too few episodes, or q95 == q05 (never replaced by zero)."""


def fit_scalers(columns: dict, min_episodes: int = 200, quantiles=(0.05, 0.95), upper_ladder=(0.95, 0.99, 1.0)) -> dict:
    """Fit {metric: {q05, q95, upper_quantile, n}} from arrays of successful quality-valid episodes of ONE backend.

    The lower anchor is q05. The upper anchor is the first quantile of `upper_ladder` that exceeds it (q95, then q99, then
    the observed maximum): several descriptors are zero-anchored (a clean reach has no extra speed peak, no pause and no joint
    reversal), so q05 and q95 can coincide while the metric still varies in the tail. `q95` in the result is the effective
    upper anchor and `upper_quantile` records which rung was used. A metric that never varies on the calibration data carries
    no information and remains a blocking ScalerError, never silently replaced by zero.
    """
    scalers = {}
    for metric in SCALED_METRICS:
        if metric not in columns:
            raise ScalerError(f'metric {metric!r} missing from calibration data')
        values = np.asarray(columns[metric], dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) < min_episodes:
            raise ScalerError(f'{metric}: {len(values)} valid calibration episodes < required {min_episodes}')
        lo = float(np.quantile(values, quantiles[0]))
        for rung in upper_ladder:
            hi = float(np.quantile(values, rung))
            if hi > lo:
                break
        else:
            raise ScalerError(f'degenerate scale: {metric} is constant ({lo!r}) on all {len(values)} calibration episodes; '
                              'it carries no information and is not replaced by zero (blocking calibration error)')
        scalers[metric] = {'q05': lo, 'q95': hi, 'upper_quantile': float(rung), 'n': int(len(values)),
                           'direction': SCALED_METRICS[metric]}
    return scalers


def b_up(x, lo, hi):
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def b_down(x, lo, hi):
    return 1.0 - b_up(x, lo, hi)


def _bounded(metric, values, scalers):
    s = scalers[metric]
    fn = b_up if SCALED_METRICS[metric] == 'up' else b_down
    return fn(values, s['q05'], s['q95'])


def score_arrays(metrics: dict, scalers: dict) -> dict:
    """Vectorised domain scores and HKS for arrays of per-episode raw metrics (NaN stays NaN)."""
    out = {}
    total = 0.0
    for domain, parts in DOMAINS.items():
        value = sum(w * _bounded(m, metrics[m], scalers) for m, w in parts)
        out[domain] = value
        total = total + value
    out['HKS'] = 25.0 * total
    return out


def score_episode(metrics: dict, scalers: dict) -> dict:
    """Domain scores and HKS for one episode (floats); non-finite inputs give NaN and valid=False."""
    arrays = {k: np.asarray([metrics[k]], dtype=np.float64) for k in SCALED_METRICS}
    scored = {k: float(v[0]) for k, v in score_arrays(arrays, scalers).items()}
    scored['valid'] = bool(math.isfinite(scored['HKS']))
    return scored


def r_hks(terminal_hold, hks, valid=None):
    """Reliability-adjusted score per episode: zero unless a terminal hold AND a finite, valid HKS.

    Returns (R_HKS array, unscorable_success mask): a claimed success with a missing/invalid HKS is flagged,
    scored zero, and counted by the caller so the quality-failure count is always shown next to the score.
    """
    hold = np.asarray(terminal_hold, dtype=bool)
    hks = np.asarray(hks, dtype=np.float64)
    ok = np.isfinite(hks) if valid is None else (np.isfinite(hks) & np.asarray(valid, dtype=bool))
    unscorable = hold & ~ok
    return np.where(hold & ok, hks, 0.0), unscorable
