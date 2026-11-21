"""Outcome and kinematic-descriptor tests on analytic trajectories with known answers."""
import numpy as np
import pytest

from natural_rl.metrics.kinematics import (KinematicsConfig, joint_reversal_rate, min_jerk_speed, savgol_derivatives,
                                           segment_kinematics, sparc, template_rmse)
from natural_rl.metrics.outcomes import episode_outcomes

from .conftest import concatenate_reaches, min_jerk_position, min_jerk_reach

START, GOAL = np.array([0.30, -0.10, 0.10]), np.array([0.55, 0.05, 0.20])    # 0.3 m reach


def kin(ee, q, goal=GOAL, hold=True, cfg=None, **kwargs):
    return segment_kinematics(ee, q, goal, hold, cfg or KinematicsConfig(), **kwargs)


# ------------------------------------------------------------------ outcomes
def test_terminal_hold_needs_the_last_five_samples_inside():
    goal = np.zeros(3)
    inside, outside = np.zeros(3), np.array([0.05, 0, 0])
    four = np.array([outside] * 10 + [inside] * 4)
    five = np.array([outside] * 10 + [inside] * 5)
    assert not episode_outcomes(four, goal, 0.05)['terminal_hold']
    assert episode_outcomes(five, goal, 0.05)['terminal_hold']


def test_first_entry_any_hold_and_terminal_hold_are_separate_events():
    goal = np.zeros(3)
    path = np.array([[0.1, 0, 0]] * 3 + [[0.0, 0, 0]] * 6 + [[0.1, 0, 0]] * 4 + [[0.0, 0, 0]] * 2)
    out = episode_outcomes(path, goal, 0.05)
    assert out['first_entry'] and np.isclose(out['first_entry_s'], 3 * 0.05)
    assert out['any_hold'] and not out['terminal_hold']          # held for 6 samples, then left and ended outside the 5-window
    assert out['goal_departures'] == 1
    assert np.isclose(out['final_error_m'], 0.0)


def test_first_entry_ignores_the_initial_sample():
    goal = np.zeros(3)
    path = np.array([[0.0, 0, 0]] + [[0.2, 0, 0]] * 5)           # starts inside, never accepted inside afterwards
    out = episode_outcomes(path, goal, 0.05)
    assert not out['first_entry'] and np.isnan(out['first_entry_s'])


def test_early_episode_end_is_not_padded():
    out = episode_outcomes(np.zeros((4, 3)), np.zeros(3), 0.05)
    assert out['n_decisions'] == 3 and not out['terminal_hold']   # fewer than 5 samples cannot be a 5-step hold


# ------------------------------------------------------------------ kinematic descriptors
def test_minimum_jerk_reach_has_one_peak_and_matches_the_template():
    ee, q = min_jerk_reach(START, GOAL, duration=1.6)
    m = kin(ee, q)
    assert m['valid'] and m['quality_flag'] == ''
    assert m['speed_peaks'] == 1 and m['peaks_excess'] == 0
    assert m['pause_fraction'] < 0.1 and m['joint_reversal_rate'] == 0.0
    assert 0.98 < m['path_efficiency'] < 1.05                     # straight path


def ideal_template_floor(fraction=0.05):
    """Template RMSE of an IDEAL minimum-jerk reach scored on the window where speed >= `fraction` of its peak.

    It is not zero: the window drops the first and last ~6 % of the movement, so the stretched profile differs a
    little from 30 s^2 (1-s)^2. Every policy is scored with the same window rule; this is the floor.
    """
    tau = np.linspace(0, 1, 2_000_001)
    inside = np.flatnonzero(min_jerk_speed(tau) >= fraction * min_jerk_speed(tau).max())
    a, b = tau[inside[0]], tau[inside[-1]]
    s = np.linspace(0, 1, 101)
    window = min_jerk_speed(a + s * (b - a))
    return float(np.sqrt(np.mean((window / np.trapz(window, s) - min_jerk_speed(s)) ** 2)))


def test_template_error_of_an_ideal_reach_matches_its_analytic_floor_and_beats_clearly_different_profiles():
    floor = ideal_template_floor()
    assert 0.13 < floor < 0.16
    ee, q = min_jerk_reach(START, GOAL, duration=2.4)             # finer sampling: discretisation error is small
    ideal = kin(ee, q)['template_rmse']
    assert np.isclose(ideal, floor, rtol=0.08)
    # a constant-speed movement is far from the template; so is a movement with a pause in the middle
    n = 48
    ee_c = START + np.linspace(0, 1, n)[:, None] * (GOAL - START)
    ee_c = np.concatenate([np.tile(START, (4, 1)), ee_c, np.tile(GOAL, (12, 1))])
    assert kin(ee_c, np.zeros((len(ee_c), 7)))['template_rmse'] > 3 * ideal
    mid = (START + GOAL) / 2
    ee_p, q_p = concatenate_reaches([START, mid, GOAL], [1.2, 1.2], pause_samples=8)
    assert kin(ee_p, q_p)['template_rmse'] > ideal


def test_dimensionless_jerk_matches_the_analytic_value_over_the_scored_window():
    duration, dt = 3.0, 0.05
    ee, q = min_jerk_reach(START, GOAL, duration, dt)
    m = kin(ee, q)
    t_on, t_off = m['onset_k'] * dt, m['offset_k'] * dt
    # Analytic DJ of the continuous minimum-jerk motion over exactly the scored window [t_on, t_off].
    lead = 4 * dt
    t = np.linspace(t_on - lead, t_off - lead, 200001)            # movement-local time
    tau = t / duration
    distance = np.linalg.norm(GOAL - START)
    jerk = distance / duration ** 3 * (60 - 360 * tau + 360 * tau ** 2)
    length = distance * (min_jerk_position(tau[-1]) - min_jerk_position(tau[0]))
    window = t[-1] - t[0]
    expected = window ** 5 / length ** 2 * np.trapz(jerk ** 2, t)
    assert np.isclose(m['dimensionless_jerk'], expected, rtol=0.05)
    # Validate the analytic reference itself: over the FULL movement the same formula gives the textbook 720 ...
    tau_full = np.linspace(0, 1, 200001)
    jerk_full = distance / duration ** 3 * (60 - 360 * tau_full + 360 * tau_full ** 2)
    full = duration ** 5 / distance ** 2 * np.trapz(jerk_full ** 2, tau_full * duration)
    assert np.isclose(full, 720.0, rtol=1e-4)
    # ... and the 5 %-speed scoring window removes the high-jerk start and end, so an ideal reach scores ~200, a
    # constant, policy-independent floor (log1p ~ 5.3) that the figures show as the ideal reference.
    assert 150 < m['dimensionless_jerk'] < 250


def test_pause_between_sub_movements_costs_smoothness_and_adds_a_peak():
    mid = (START + GOAL) / 2 + np.array([0.0, 0.0, 0.0])
    single_ee, single_q = min_jerk_reach(START, GOAL, 2.0)
    split_ee, split_q = concatenate_reaches([START, mid, GOAL], [1.0, 1.0], pause_samples=8)
    single, split = kin(single_ee, single_q), kin(split_ee, split_q)
    assert split['speed_peaks'] == 2 and split['peaks_excess'] == 1
    assert split['pause_fraction'] > 0.1 > single['pause_fraction']
    assert split['sparc'] < single['sparc']                       # more negative = less smooth
    assert split['template_rmse'] > single['template_rmse']
    assert split['valid'] and single['valid']


def test_a_trailing_second_movement_is_not_hidden_by_cutting_at_the_first_pause():
    """Offset is the LAST active sample before the terminal window, so late sub-movements are still scored."""
    mid = (START + GOAL) / 2
    ee, q = concatenate_reaches([START, mid, GOAL], [1.0, 1.0], pause_samples=10)
    m = kin(ee, q)
    assert m['offset_k'] > 4 + 20 + 10                            # beyond the end of the first sub-movement and the pause


def test_the_steady_hold_never_enters_the_scored_segment():
    ee1, q1 = min_jerk_reach(START, GOAL, 1.6, hold=8)
    ee2, q2 = min_jerk_reach(START, GOAL, 1.6, hold=60)
    a, b = kin(ee1, q1), kin(ee2, q2)
    for key in ('segment_duration_s', 'path_efficiency', 'log1p_dj', 'sparc', 'template_rmse', 'pause_fraction'):
        assert np.isclose(a[key], b[key], rtol=1e-9, atol=1e-12), key


def test_terminal_run_start_rule_truncates_an_ideal_reach_which_is_why_it_is_not_primary():
    ee, q = min_jerk_reach(START, GOAL, 2.0)
    primary = kin(ee, q)
    truncated = kin(ee, q, cfg=KinematicsConfig(offset_rule='terminal_run_start'))
    assert truncated['segment_duration_s'] < 0.9 * primary['segment_duration_s']


def test_zero_displacement_is_flagged_and_never_scored_as_perfectly_smooth():
    still = np.tile(START, (40, 1))
    m = kin(still, np.zeros((40, 7)), goal=START)
    assert not m['valid'] and m['quality_flag'] == 'zero_displacement'
    assert np.isnan(m['log1p_dj']) and np.isnan(m['sparc'])


def test_short_and_nonfinite_signals_are_flagged():
    short = kin(np.zeros((5, 3)), np.zeros((5, 7)))
    assert not short['valid'] and short['quality_flag'] == 'short_segment'
    ee, q = min_jerk_reach(START, GOAL, 1.6)
    ee[10, 0] = np.nan
    bad = kin(ee, q)
    assert not bad['valid'] and bad['quality_flag'] == 'nonfinite'


def test_episode_without_terminal_hold_is_flagged_for_diagnostics_only():
    ee, q = min_jerk_reach(START, START + 0.5 * (GOAL - START), 1.6)       # stops halfway
    m = kin(ee, q, hold=False)
    assert m['quality_flag'] == 'no_terminal_hold' and not m['valid']


# ------------------------------------------------------------------ SPARC, template, reversals, derivatives
def test_sparc_rejects_cutoffs_at_or_above_nyquist():
    speed = min_jerk_speed(np.linspace(0, 1, 30))
    with pytest.raises(ValueError):
        sparc(speed, fs=20.0, cutoff_hz=10.0)
    with pytest.raises(ValueError):
        sparc(speed, fs=20.0, cutoff_hz=20.0)
    assert np.isfinite(sparc(speed, fs=20.0, cutoff_hz=8.0))


def test_sparc_is_less_negative_for_a_single_smooth_bell_than_for_a_jagged_profile():
    s = np.linspace(0, 1, 41)
    smooth = min_jerk_speed(s)
    jagged = smooth * (1 + 0.4 * np.sin(2 * np.pi * 9 * s))
    assert sparc(smooth, 20.0) > sparc(jagged, 20.0)


def test_template_rmse_is_near_zero_for_the_template_and_large_for_a_flat_profile():
    s = np.linspace(0, 1, 101)
    assert template_rmse(min_jerk_speed(s)) < 1e-3
    assert template_rmse(np.ones(101)) > 0.5


def test_joint_reversals_count_sign_changes_and_ignore_sub_threshold_motion():
    q = np.zeros((9, 2))
    q[:, 0] = [0, 1, 2, 1, 0, 1, 2, 3, 4]                         # +,+,-,-,+,+,+,+  -> 2 reversals
    q[:, 1] = np.arange(9) * 1e-6                                  # below threshold: no reversal, no motion
    assert np.isclose(joint_reversal_rate(q, 0.05, 1e-4), 2 / (0.05 * 8))
    assert joint_reversal_rate(np.cumsum(np.ones((9, 2)), axis=0), 0.05, 1e-4) == 0.0


def test_savgol_derivatives_recover_a_cubic_exactly():
    dt = 0.05
    t = np.arange(0, 2, dt)
    p = np.stack([t ** 3, 2 * t ** 2, t], axis=1)
    v, a, j = savgol_derivatives(p, dt)
    np.testing.assert_allclose(v[:, 0], 3 * t ** 2, atol=1e-9)
    np.testing.assert_allclose(j[:, 0], np.full_like(t, 6.0), atol=1e-6)
    np.testing.assert_allclose(a[:, 1], np.full_like(t, 4.0), atol=1e-8)
