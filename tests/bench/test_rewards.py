"""Reward terms: historical r0 parity, hold/dwell indexing, acceleration history, ROM, step-vs-history equality,
relabelled-goal recomputation, profile composition and the calibration lock."""
import copy
import os

import numpy as np
import pytest
import yaml

from natural_rl.rewards import RewardNotLocked, RewardProfile, history_terms, initial_state, load_profile, step_terms
from natural_rl.rewards.terms import RewardConfig, _r0, term_bounds

from .conftest import ROOT

J = 7
LOWER, UPPER = np.full(J, -1.0), np.full(J, 1.0)
CFG = RewardConfig()


@pytest.fixture(scope='module')
def reward_yaml():
    with open(os.path.join(ROOT, 'configs', 'rewards', 'reward_v1.yaml')) as handle:
        return yaml.safe_load(handle)


def locked(cfg, value=2.0):
    cfg = copy.deepcopy(cfg)
    cfg['status'] = 'locked'
    cfg['calibration']['lambda'] = {k: value for k in cfg['calibration']['lambda']}
    return cfg


def run_steps(goal, p, v, q, collided):
    """Roll `step_terms` over (n, H) arrays and stack the per-step outputs into (n, H) histories."""
    n, h = p.shape[0], p.shape[1]
    state = initial_state(n, J)
    p_prev = np.concatenate([np.zeros((n, 1, 3)), p[:, :-1]], axis=1)
    out = {}
    for t in range(h):
        terms, state = step_terms(state, p_prev[:, t], p[:, t], goal, v[:, t], q[:, t], collided[:, t], LOWER, UPPER, CFG)
        terms['count'], terms['held_once'] = state['count'], state['held_once']
        for key, value in terms.items():
            out.setdefault(key, []).append(value)
    return {k: np.stack(v, axis=1) for k, v in out.items()}, p_prev


def random_episode(rng, n=6, h=40, goal_scale=0.03):
    goal = rng.uniform(-0.5, 0.5, (n, 3))
    # wander near the goal so that dwell, exits and re-entries all occur
    p = goal[:, None, :] + rng.normal(0, goal_scale, (n, h, 3)) * rng.choice([0.2, 1.0], (n, h, 1))
    v = rng.choice([-1, -0.3, -0.1, 0, 0.1, 0.3, 1.0], size=(n, h, J))
    q = rng.uniform(-1, 1, (n, h, J))
    collided = rng.random((n, h)) < 0.1
    return goal, p, v, q, collided


# ------------------------------------------------------------------ historical r0
def test_r0_matches_the_historical_implementation_exactly():
    mdp = pytest.importorskip('kangaroo_rl.mdp')
    rng = np.random.default_rng(0)
    historical = mdp.Reward()
    point = rng.normal(0, 0.4, (500, 3))
    goal = rng.normal(0, 0.4, (500, 3))
    collided = rng.random(500) < 0.3
    velocity = rng.choice([-1, -0.3, -0.1, 0, 0.1, 0.3, 1.0], size=(500, J))
    d = np.linalg.norm(point - goal, axis=-1)
    np.testing.assert_allclose(_r0(d, collided, velocity, CFG), historical(point, goal, collided, velocity), rtol=0, atol=1e-12)


def test_r0_special_values():
    assert np.isclose(_r0(np.array(0.0), False, np.zeros(J), CFG), 1.0)             # at the goal: bonus 1, no cost
    assert np.isclose(_r0(np.array(2.0), False, np.zeros(J), CFG), -1.0)            # distance term saturates
    assert np.isclose(_r0(np.array(0.0), True, np.zeros(J), CFG), 0.0)              # collision flag costs exactly 1
    assert np.isclose(_r0(np.array(0.0), False, np.ones(J), CFG), 1.0 - 0.01)       # u = mean((v / vmax)^2) = 1


# ------------------------------------------------------------------ hold and time
def inside_pattern(pattern):
    """A single episode whose accepted position is on the goal (1) or 5 cm away (0) at each step."""
    goal = np.zeros((1, 3))
    p = np.array([[0.0, 0, 0] if flag else [0.05, 0, 0] for flag in pattern])[None]
    n, h = 1, len(pattern)
    return goal, p, np.zeros((n, h, J)), np.zeros((n, h, J)), np.zeros((n, h), dtype=bool)


def test_dwell_counter_counts_consecutive_inside_steps_and_resets_on_exit():
    pattern = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    out, _ = run_steps(*inside_pattern(pattern))
    np.testing.assert_array_equal(out['count'][0], [0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2])
    # 0.2 whenever inside; a further 0.8 at every step once the counter has reached 5 (steps 6, 7, 8)
    np.testing.assert_allclose(out['hold'][0], [0, 0, .2, .2, .2, .2, 1.0, 1.0, 1.0, 0, .2, .2])


def test_time_penalty_runs_until_the_step_where_the_first_hold_completes_and_never_returns():
    pattern = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    out, _ = run_steps(*inside_pattern(pattern))
    step_cost = -0.05 / 5.0
    expected = [step_cost] * 7 + [0.0] * 5                                      # penalty at steps 0..6 (first hold at 6)
    np.testing.assert_allclose(out['time'][0], expected)
    assert out['held_once'][0].tolist() == [False] * 6 + [True] * 6               # stays set after leaving tolerance


# ------------------------------------------------------------------ other terms
def test_acceleration_uses_the_previous_command_and_starts_from_rest():
    goal = np.zeros((1, 3))
    p = np.zeros((1, 3, 3))
    v = np.zeros((1, 3, J))
    v[0, 0], v[0, 1], v[0, 2] = 1.0, 1.0, -1.0
    out, _ = run_steps(goal, p, v, np.zeros((1, 3, J)), np.zeros((1, 3), dtype=bool))
    a = 1.0 / (0.05 * 40.0)                                                      # command jump of v_max per decision
    np.testing.assert_allclose(out['accel'][0], [-(a ** 2), 0.0, -((2 * a) ** 2)])


def test_path_term_is_executed_end_effector_motion_over_l_ref():
    goal = np.zeros((1, 3))
    p = np.array([[[0.1, 0, 0], [0.1, 0.2, 0.0]]])
    out, p_prev = run_steps(goal, p, np.zeros((1, 2, J)), np.zeros((1, 2, J)), np.zeros((1, 2), dtype=bool))
    # step 0 starts from the supplied p_prev (origin in this helper): ||0.1|| / 0.5 ; step 1: ||0.2|| / 0.5
    np.testing.assert_allclose(out['path'][0], [-0.1 / 0.5, -0.2 / 0.5])


def test_rom_is_zero_in_the_middle_of_the_range_and_one_joint_at_a_limit_costs_one_seventh():
    goal = np.zeros((1, 3))
    middle = np.zeros((1, 1, J))
    at_limit = np.zeros((1, 1, J))
    at_limit[0, 0, 0] = 1.0                                                      # joint 0 exactly at its upper limit
    p = np.zeros((1, 1, 3))
    none = np.zeros((1, 1), dtype=bool)
    assert run_steps(goal, p, np.zeros((1, 1, J)), middle, none)[0]['rom'][0, 0] == 0.0
    assert np.isclose(run_steps(goal, p, np.zeros((1, 1, J)), at_limit, none)[0]['rom'][0, 0], -1.0 / J)


def test_speed_term_is_time_weighted_normalised_squared_command():
    goal = np.zeros((1, 3))
    out, _ = run_steps(goal, np.zeros((1, 1, 3)), np.ones((1, 1, J)), np.zeros((1, 1, J)), np.zeros((1, 1), dtype=bool))
    assert np.isclose(out['speed'][0, 0], -(0.05 / 5.0))


# ------------------------------------------------------------------ stepwise == history
@pytest.mark.parametrize('seed', range(5))
def test_stepwise_and_history_implementations_agree_on_every_term_and_feature(seed):
    goal, p, v, q, collided = random_episode(np.random.default_rng(seed))
    stepped, p_prev = run_steps(p=p, goal=goal, v=v, q=q, collided=collided)
    hist = history_terms(p_prev, p, goal, v, q, collided, LOWER, UPPER, CFG)
    for key in ('r0', 'hold', 'time', 'path', 'speed', 'accel', 'rom', 'd', 'count', 'held_once'):
        np.testing.assert_allclose(stepped[key], hist[key], atol=1e-12, err_msg=key)
    assert (hist['count'] > 4).any() and (hist['count'] == 0).any()               # the test really exercised dwell logic


def test_relabelled_goal_recomputes_dwell_and_first_hold_over_the_whole_history():
    """Replacing the goal by a point the arm later reached must change the counters of EARLIER steps too."""
    rng = np.random.default_rng(11)
    goal, p, v, q, collided = random_episode(rng, n=1, h=40, goal_scale=0.2)      # far from the original goal
    p_prev = np.concatenate([np.zeros((1, 1, 3)), p[:, :-1]], axis=1)
    original = history_terms(p_prev, p, goal, v, q, collided, LOWER, UPPER, CFG)
    # relabel: make the arm sit exactly on the new goal for 6 steps from step 10 on
    p2 = p.copy()
    new_goal = np.array([[0.1, 0.2, 0.3]])
    p2[0, 10:16] = new_goal
    p2_prev = np.concatenate([np.zeros((1, 1, 3)), p2[:, :-1]], axis=1)
    relabelled = history_terms(p2_prev, p2, new_goal, v, q, collided, LOWER, UPPER, CFG)
    np.testing.assert_array_equal(relabelled['count'][0, 10:16], [1, 2, 3, 4, 5, 6])
    assert relabelled['held_once'][0, 14:].all() and not relabelled['held_once'][0, :14].any()
    # the penalty runs through the step at which the first hold completes (14) and never returns
    assert (relabelled['time'][0, :15] < 0).all() and (relabelled['time'][0, 15:] == 0).all()
    assert (original['count'] < 5).all()


# ------------------------------------------------------------------ profiles
EXPECTED = {
    'r0_original': ['r0'], 'r1_hold': ['r0', 'hold'], 'r2_time': ['r0', 'hold', 'time'], 'r3_path': ['r0', 'hold', 'path'],
    'r4_speed': ['r0', 'hold', 'speed'], 'r5_smooth': ['r0', 'hold', 'accel'], 'r6_rom': ['r0', 'hold', 'rom'],
    'r7_balanced': ['r0', 'hold', 'path', 'speed', 'accel', 'rom'],
}


def test_all_twelve_profiles_are_registered_with_the_protocol_terms(reward_yaml):
    assert len(reward_yaml['profiles']) == 12
    for name, terms in EXPECTED.items():
        assert sorted(reward_yaml['profiles'][name]['terms']) == sorted(terms), name
    r7 = set(reward_yaml['profiles']['r7_balanced']['terms'])
    for drop, term in {'r8_drop_path': 'path', 'r9_drop_speed': 'speed', 'r10_drop_smooth': 'accel', 'r11_drop_rom': 'rom'}.items():
        assert set(reward_yaml['profiles'][drop]['terms']) == r7 - {term}, drop


def test_profiles_with_calibrated_terms_refuse_to_build_until_the_register_is_locked(reward_yaml):
    reward_yaml = copy.deepcopy(reward_yaml)                                       # the shipped register may already be locked
    reward_yaml['status'] = 'unlocked'
    reward_yaml['calibration']['lambda'] = {k: None for k in reward_yaml['calibration']['lambda']}
    load_profile(reward_yaml, 'r0_original')                                       # no calibrated term
    load_profile(reward_yaml, 'r1_hold')                                           # hold weights are fixed, not calibrated
    for name in ('r2_time', 'r3_path', 'r4_speed', 'r5_smooth', 'r6_rom', 'r7_balanced', 'r8_drop_path', 'r11_drop_rom'):
        with pytest.raises(RewardNotLocked, match='calibrate'):
            load_profile(reward_yaml, name)
    assert isinstance(load_profile(reward_yaml, 'r7_balanced', allow_unlocked=True), RewardProfile)


def test_lock_requires_a_lambda_for_every_calibrated_term_in_the_profile(reward_yaml):
    cfg = locked(reward_yaml)
    cfg['calibration']['lambda']['path'] = None
    with pytest.raises(RewardNotLocked):
        load_profile(cfg, 'r3_path')
    load_profile(cfg, 'r4_speed')                                                  # unaffected terms still load


def test_weighted_terms_and_total_follow_r_equals_r0_plus_sum_lambda_f(reward_yaml):
    cfg = locked(reward_yaml)
    cfg['calibration']['lambda'].update(time=1.5, path=2.0, speed=3.0, accel=4.0, rom=5.0)
    profile = load_profile(cfg, 'r7_balanced')
    goal, p, v, q, collided = random_episode(np.random.default_rng(3), n=2, h=10)
    p_prev = np.concatenate([np.zeros((2, 1, 3)), p[:, :-1]], axis=1)
    raw = history_terms(p_prev, p, goal, v, q, collided, LOWER, UPPER, CFG)
    total, weighted = profile.combine(raw)
    assert set(weighted) == {'r0', 'hold', 'path', 'speed', 'accel', 'rom'}       # 'time' is not in r7
    np.testing.assert_allclose(weighted['path'], 2.0 * raw['path'])
    np.testing.assert_allclose(weighted['accel'], 4.0 * raw['accel'])
    np.testing.assert_allclose(total, raw['r0'] + raw['hold'] + 2 * raw['path'] + 3 * raw['speed'] + 4 * raw['accel'] + 5 * raw['rom'])


def test_profile_bounds_contain_every_reachable_reward(reward_yaml):
    cfg = locked(reward_yaml, value=2.0)
    rng = np.random.default_rng(4)
    for name in ('r0_original', 'r1_hold', 'r7_balanced'):
        profile = load_profile(cfg, name)
        low, high = profile.bounds()
        for seed in range(3):
            goal, p, v, q, collided = random_episode(rng, n=4, h=30, goal_scale=0.5)
            p = np.clip(p, -0.04, 0.04)                                              # every step stays below max_ee_step = 0.15 m
            p_prev = np.concatenate([np.zeros((4, 1, 3)), p[:, :-1]], axis=1)
            raw = history_terms(p_prev, p, goal, v, q, collided, LOWER, UPPER, CFG)
            total, _ = profile.combine(raw)
            assert (total >= low - 1e-9).all() and (total <= high + 1e-9).all(), name
    np.testing.assert_allclose(load_profile(cfg, 'r0_original').bounds(), (-2.01, 1.0))   # the historical bounds
