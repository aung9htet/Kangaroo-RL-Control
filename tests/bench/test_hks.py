"""HKS scaling, domain scoring and the reliability-adjusted R-HKS (failures contribute zero)."""
import numpy as np
import pytest

from natural_rl.metrics.hks import (DOMAINS, SCALED_METRICS, ScalerError, b_down, b_up, fit_scalers, r_hks,
                                    score_arrays, score_episode)

SCALERS = {m: {'q05': 0.0, 'q95': 1.0, 'n': 500, 'direction': d} for m, d in SCALED_METRICS.items()}


def episode(**overrides):
    """An episode at the *best* end of every scale unless overridden."""
    best = {m: (1.0 if d == 'up' else 0.0) for m, d in SCALED_METRICS.items()}
    best.update(overrides)
    return best


def test_bounds_clip_and_directions():
    assert b_up(-5, 0, 1) == 0 and b_up(5, 0, 1) == 1 and b_up(0.25, 0, 1) == 0.25
    assert b_down(-5, 0, 1) == 1 and b_down(5, 0, 1) == 0 and b_down(0.25, 0, 1) == 0.75


def test_best_and_worst_episodes_score_100_and_0():
    assert np.isclose(score_episode(episode(), SCALERS)['HKS'], 100.0)
    worst = {m: (0.0 if d == 'up' else 1.0) for m, d in SCALED_METRICS.items()}
    assert np.isclose(score_episode(worst, SCALERS)['HKS'], 0.0)
    mid = {m: 0.5 for m in SCALED_METRICS}
    assert np.isclose(score_episode(mid, SCALERS)['HKS'], 50.0)


def test_every_domain_has_the_same_25_point_weight():
    scored = score_episode(episode(), SCALERS)
    for domain in DOMAINS:
        assert np.isclose(25.0 * scored[domain], 25.0)
    # Within a domain the correlated descriptors share weight instead of counting twice.
    only_sparc_bad = score_episode(episode(sparc=0.0), SCALERS)
    assert np.isclose(only_sparc_bad['HKS'], 100.0 - 12.5)


def test_nonfinite_input_gives_nan_and_invalid_not_a_perfect_score():
    scored = score_episode(episode(sparc=float('nan')), SCALERS)
    assert not scored['valid'] and np.isnan(scored['HKS'])


def test_vectorised_scoring_matches_episode_scoring():
    cols = {m: np.array([0.2, 0.7, 0.5]) for m in SCALED_METRICS}
    vec = score_arrays(cols, SCALERS)['HKS']
    for i in range(3):
        one = score_episode({m: cols[m][i] for m in cols}, SCALERS)['HKS']
        assert np.isclose(vec[i], one)


# ------------------------------------------------------------------ scalers
def test_fit_scalers_uses_5_and_95_percent_quantiles():
    rng = np.random.default_rng(0)
    cols = {m: rng.uniform(0, 1, 1000) for m in SCALED_METRICS}
    s = fit_scalers(cols, min_episodes=200)
    for m in SCALED_METRICS:
        assert np.isclose(s[m]['q05'], np.quantile(cols[m], 0.05)) and np.isclose(s[m]['q95'], np.quantile(cols[m], 0.95))


def test_a_metric_that_never_varies_is_a_blocking_error_not_a_silent_zero():
    rng = np.random.default_rng(0)
    cols = {m: rng.uniform(0, 1, 1000) for m in SCALED_METRICS}
    cols['peaks_excess'] = np.zeros(1000)                           # constant: carries no information
    with pytest.raises(ScalerError, match='degenerate'):
        fit_scalers(cols)


def test_zero_anchored_metrics_with_a_tail_use_the_next_upper_quantile_and_record_it():
    rng = np.random.default_rng(1)
    cols = {m: rng.uniform(0, 1, 1000) for m in SCALED_METRICS}
    peaks = np.zeros(1000)
    peaks[:30] = 1.0                                                # 3 % of episodes: q95 == q05 == 0 but q99 = 1
    peaks[:3] = 4.0                                                 # 0.3 %
    cols['peaks_excess'] = peaks
    s = fit_scalers(cols)['peaks_excess']
    assert s['q05'] == 0.0 and s['q95'] == 1.0 and s['upper_quantile'] == 0.99
    only_extreme = np.zeros(1000)
    only_extreme[:2] = 3.0                                          # 0.2 %: only the maximum separates from zero
    cols['peaks_excess'] = only_extreme
    s = fit_scalers(cols)['peaks_excess']
    assert s['q95'] == 3.0 and s['upper_quantile'] == 1.0
    assert fit_scalers({**cols, 'peaks_excess': rng.uniform(0, 1, 1000)})['peaks_excess']['upper_quantile'] == 0.95


def test_too_few_calibration_episodes_is_blocking():
    cols = {m: np.linspace(0, 1, 50) for m in SCALED_METRICS}
    with pytest.raises(ScalerError, match='< required'):
        fit_scalers(cols, min_episodes=200)


def test_missing_metric_is_blocking():
    cols = {m: np.linspace(0, 1, 500) for m in SCALED_METRICS}
    del cols['sparc']
    with pytest.raises(ScalerError, match='missing'):
        fit_scalers(cols)


# ------------------------------------------------------------------ R-HKS
def test_failure_contributes_zero_even_with_a_high_hks():
    r, unscorable = r_hks([False, True], [95.0, 80.0])
    assert list(r) == [0.0, 80.0] and not unscorable.any()


def test_claimed_success_with_missing_or_invalid_hks_scores_zero_and_is_counted():
    r, unscorable = r_hks([True, True, True], [float('nan'), 70.0, 60.0], valid=[True, False, True])
    assert list(r) == [0.0, 0.0, 60.0]
    assert list(unscorable) == [True, True, False]


def test_r_hks_is_the_mean_over_all_registered_tasks():
    hold = [True, False, True, False]
    hks = [80.0, 99.0, 60.0, 10.0]
    r, _ = r_hks(hold, hks)
    assert np.isclose(r.mean(), (80 + 60) / 4)                      # failures stay in the denominator
