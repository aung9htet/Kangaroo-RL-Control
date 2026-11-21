"""Hierarchical (crossed task x seed) bootstrap, rank probabilities and Benjamini-Hochberg correction.

A trajectory frame is not an independent replicate. Every uncertainty statement resamples the two real
experimental units, held-out task IDs and training seeds, and never frames or individual steps. Model
comparisons use the same task IDs for every model; paired effects use the same (seed, task) cells for baseline
and treatment. Missing cells (crashes, timeouts) are excluded and counted, never replaced by a mean.
"""
from __future__ import annotations

import numpy as np


def _weights(rng, n, resamples):
    """(resamples, n) multinomial draw counts of n units, as fractions of n."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=resamples).astype(np.float64)


def _cell_mean(w_seed, w_task, values):
    """Weighted mean of values[s, t] under per-replicate seed/task weights, ignoring NaN cells.

    Returns one estimate per replicate; replicates with no observed cell are NaN.
    """
    observed = np.isfinite(values)
    filled = np.where(observed, values, 0.0)
    numerator = np.einsum('bs,st,bt->b', w_seed, filled, w_task, optimize=True)
    denominator = np.einsum('bs,st,bt->b', w_seed, observed.astype(np.float64), w_task, optimize=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return numerator / denominator


def point_estimate(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')


def interval(replicates, level=0.95):
    lo, hi = np.nanquantile(replicates, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def bootstrap_leaderboard(scores: dict, resamples=2000, seed=20261121, level=0.95) -> dict:
    """Mean score, interval and rank probabilities for several models.

    scores: {model: array (n_seeds_m, n_tasks)} of per-(seed, task) values (R-HKS, failures = 0, missing = NaN).
    All models must share the same task axis (same task IDs, same order). Seeds are resampled independently
    per model; tasks are resampled jointly across models, so the rank statistics respect task pairing.
    Returns {'models': [...], 'mean': ..., 'ci_low': ..., 'ci_high': ..., 'rank_probability': (M, M) with
    column k-1 = P(rank = k), 'p_best': P(rank = 1), 'n_seeds': ..., 'n_tasks': ...}.
    """
    models = list(scores)
    n_tasks = {np.asarray(v).shape[1] for v in scores.values()}
    if len(n_tasks) != 1:
        raise ValueError(f'models disagree on the task axis: {sorted(n_tasks)}')
    n_tasks = n_tasks.pop()
    rng = np.random.default_rng(seed)
    w_task = _weights(rng, n_tasks, resamples)
    replicates = np.empty((len(models), resamples))
    for i, model in enumerate(models):
        values = np.asarray(scores[model], dtype=np.float64)
        replicates[i] = _cell_mean(_weights(rng, values.shape[0], resamples), w_task, values)
    # Rank 1 = highest score; ties share the best (lowest) rank, which only matters for exact ties.
    order = (-replicates).argsort(axis=0, kind='stable')
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(len(models))[:, None].repeat(resamples, axis=1), axis=0)
    probability = np.stack([(ranks == k).mean(axis=1) for k in range(len(models))], axis=1)
    ci = np.array([interval(r, level) for r in replicates])
    return {
        'models': models,
        'mean': np.array([point_estimate(scores[m]) for m in models]),
        'ci_low': ci[:, 0], 'ci_high': ci[:, 1],
        'rank_probability': probability, 'p_best': probability[:, 0],
        'n_seeds': np.array([np.asarray(scores[m]).shape[0] for m in models]),
        'n_tasks': n_tasks, 'resamples': resamples,
    }


def paired_effect(baseline, treatment, resamples=2000, seed=20261121, level=0.95) -> dict:
    """Within-policy paired effect: treatment - baseline on matched (seed, task) cells.

    Both arrays are (n_seeds, n_tasks) with identical seed and task identifiers in the same order; a cell
    missing in either is dropped from both. Returns the mean delta, its interval and a two-sided bootstrap p.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    if baseline.shape != treatment.shape:
        raise ValueError(f'unpaired inputs: {baseline.shape} vs {treatment.shape}; seed and task axes must match')
    delta = treatment - baseline                       # NaN wherever either side is missing
    rng = np.random.default_rng(seed)
    w_seed, w_task = _weights(rng, delta.shape[0], resamples), _weights(rng, delta.shape[1], resamples)
    reps = _cell_mean(w_seed, w_task, delta)
    reps = reps[np.isfinite(reps)]
    low, high = interval(reps, level)
    # Two-sided percentile-bootstrap p-value with the +1 correction (never exactly zero).
    below, above = np.sum(reps <= 0), np.sum(reps >= 0)
    p = min(1.0, 2.0 * (min(below, above) + 1) / (len(reps) + 1))
    return {'delta': point_estimate(delta), 'ci_low': low, 'ci_high': high, 'p_value': float(p),
            'n_pairs': int(np.isfinite(delta).sum()), 'resamples': resamples}


def benjamini_hochberg(p_values) -> np.ndarray:
    """BH-adjusted q-values for a registered family of tests (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    n = len(p)
    adjusted = p[order] * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(adjusted, 1.0)
    return q
