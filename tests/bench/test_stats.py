"""Paired hierarchical bootstrap alignment, rank probabilities and Benjamini-Hochberg correction."""
import numpy as np
import pytest

from natural_rl.stats import benjamini_hochberg, bootstrap_leaderboard, paired_effect


def test_benjamini_hochberg_matches_a_hand_computed_example():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_bh_is_monotone_and_capped():
    q = benjamini_hochberg([0.9, 0.5, 0.0001, 0.7])
    assert (q <= 1.0).all() and (q >= 0).all()
    order = np.argsort([0.9, 0.5, 0.0001, 0.7])
    assert (np.diff(q[order]) >= -1e-12).all()


def test_paired_effect_refuses_unaligned_inputs():
    with pytest.raises(ValueError, match='unpaired'):
        paired_effect(np.zeros((3, 10)), np.zeros((4, 10)))


def test_paired_effect_recovers_a_constant_difference():
    rng = np.random.default_rng(1)
    base = rng.normal(40, 10, (5, 60))
    out = paired_effect(base, base + 3.0, resamples=500)
    assert np.isclose(out['delta'], 3.0)
    assert out['ci_low'] > 2.99 and out['ci_high'] < 3.01
    assert out['n_pairs'] == 300 and out['p_value'] < 0.01


def test_paired_effect_of_identical_arrays_is_exactly_zero_with_a_wide_p():
    base = np.random.default_rng(2).normal(size=(5, 40))
    out = paired_effect(base, base.copy(), resamples=300)
    assert out['delta'] == 0.0 and out['ci_low'] == 0.0 and out['ci_high'] == 0.0
    assert out['p_value'] > 0.9


def test_paired_effect_pairing_beats_unpaired_noise():
    """Large shared task difficulty cancels in the paired difference: the interval is tight around the true effect."""
    rng = np.random.default_rng(3)
    difficulty = rng.normal(0, 30, 80)[None, :]                    # shared by both arms (same tasks)
    base = 50 + difficulty + rng.normal(0, 1, (5, 80))
    treat = base + 2.0 + rng.normal(0, 1, (5, 80))
    out = paired_effect(base, treat, resamples=500)
    assert out['ci_high'] - out['ci_low'] < 1.0 and out['ci_low'] > 1.0


def test_missing_cells_are_dropped_from_both_arms_and_counted_not_imputed():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(5, 20))
    treat = base + 1.0
    treat[0, :] = np.nan                                            # one crashed training seed
    base[1, 3] = np.nan
    out = paired_effect(base, treat, resamples=200)
    assert out['n_pairs'] == 5 * 20 - 20 - 1
    assert np.isclose(out['delta'], 1.0)


def test_leaderboard_ranks_a_clearly_better_model_first_with_probability_one():
    rng = np.random.default_rng(5)
    scores = {'a': rng.normal(70, 3, (5, 50)), 'b': rng.normal(40, 3, (5, 50)), 'c': rng.normal(10, 3, (5, 50))}
    out = bootstrap_leaderboard(scores, resamples=400)
    assert out['models'] == ['a', 'b', 'c']
    assert out['p_best'][0] > 0.99
    np.testing.assert_allclose(out['rank_probability'].sum(axis=0), 1.0)    # each rank is taken by someone
    np.testing.assert_allclose(out['rank_probability'].sum(axis=1), 1.0)    # each model has a rank
    assert out['ci_low'][0] < out['mean'][0] < out['ci_high'][0]


def test_identical_models_do_not_produce_a_resolved_winner():
    rng = np.random.default_rng(6)
    shared = rng.normal(50, 10, (5, 80))
    out = bootstrap_leaderboard({'a': shared + rng.normal(0, 0.5, shared.shape), 'b': shared + rng.normal(0, 0.5, shared.shape)},
                                resamples=600)
    assert 0.2 < out['p_best'][0] < 0.8


def test_leaderboard_requires_a_shared_task_axis():
    with pytest.raises(ValueError, match='task axis'):
        bootstrap_leaderboard({'a': np.zeros((5, 40)), 'b': np.zeros((5, 39))})


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    scores = {'a': np.random.default_rng(7).normal(size=(5, 30)), 'b': np.random.default_rng(8).normal(size=(5, 30))}
    x, y = bootstrap_leaderboard(scores, resamples=200, seed=1), bootstrap_leaderboard(scores, resamples=200, seed=1)
    np.testing.assert_array_equal(x['ci_low'], y['ci_low'])
    np.testing.assert_array_equal(x['rank_probability'], y['rank_probability'])
