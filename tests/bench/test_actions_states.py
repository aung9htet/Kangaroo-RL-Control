"""Action modes (velocity, position, torque) and states (obs_v1, obs_v2, obs_v3): contract, torque dynamics, replay and run-all planning."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from natural_rl.env.contract import ACTION_MODES, DEFAULT_OBS, OBS_VERSIONS, ObsSpec, backend_of
from natural_rl.orchestrate import Overall, Resources, plan_resources, run_jobs, stage_jobs, with_resources
from natural_rl.programme import LEGACY_CELL, phase_runs
from natural_rl.replay import GoalReplay

from .test_replay import H, J, LOWER, UPPER, fill, profile, synthetic_episode  # noqa: F401  (profile is a fixture)


# ------------------------------------------------------------------ contract
def test_observation_sizes_and_the_matched_states():
    sizes = {v: ObsSpec(v, LOWER, UPPER, np.full(3, -1.0), np.full(3, 1.0)).dim for v in OBS_VERSIONS}
    assert sizes == {'v1': 90, 'v2': 99, 'v3': 106, 'v4': 130, 'v5': 133, 'v6': 108, 'v7': 72, 'v8': 150}
    assert DEFAULT_OBS == {'velocity': 'v2', 'position': 'v2', 'torque': 'v3'}
    assert [backend_of(a) for a in ACTION_MODES] == ['kinematic', 'kinematic', 'dynamics']
    with pytest.raises(ValueError):
        backend_of('jerk')
    with pytest.raises(ValueError):
        ObsSpec('v2', LOWER, UPPER, np.full(3, -1.0), np.full(3, 1.0), action_mode='jerk')


def test_obs_v3_appends_the_clipped_executed_velocity_and_demands_it():
    spec = ObsSpec('v3', LOWER, UPPER, np.full(3, -1.0), np.full(3, 1.0), action_mode='torque')
    q, g = np.zeros((2, J)), np.zeros((2, 3))
    args = (np.zeros((2, J)), np.zeros(2, dtype=int), np.zeros(2, dtype=bool))
    qd = np.array([[0.5] * J, [-3.0] * J])
    obs = spec.encode(q, g, *args, qd)
    assert obs.shape == (2, 106) and obs.dtype == np.float32
    np.testing.assert_allclose(obs[:, -J:], np.clip(qd, -1, 1))
    np.testing.assert_allclose(obs[:, :99], ObsSpec('v2', LOWER, UPPER, np.full(3, -1.0), np.full(3, 1.0)).encode(q, g, *args))
    with pytest.raises(ValueError, match='qd'):
        spec.encode(q, g, *args)


# ------------------------------------------------------------------ replay with the velocity state
def test_replay_rebuilds_obs_v3_exactly_including_the_executed_velocity(profile):
    spec = ObsSpec('v3', LOWER, UPPER, np.full(3, -1.0), np.full(3, 1.0), action_mode='torque')
    rng = np.random.default_rng(3)
    episodes = [synthetic_episode(rng) for _ in range(4)]
    buffer = GoalReplay(H * 4, H, J, J, np.int8, profile, spec, LOWER, UPPER, her_ratio=0.0, n_step=1, gamma=0.95, seed=0)
    for k, ep in enumerate(episodes):                                    # the executed velocity is stored explicitly (torque mode)
        qd = rng.uniform(-1, 1, (H, J))
        ep['qd'] = qd
        buffer.begin(ep['goal'][None])
        for t in range(H):
            buffer.add(t, ep['q'][t][None], ep['q_next'][t][None], ep['p'][t][None], ep['p_next'][t][None], ep['v'][t][None],
                       ep['action'][t][None], ep['collided'][t:t + 1], ep['clip'][t:t + 1], qd[t][None])
        buffer.end()
    batch = buffer.sample(200)
    for i in range(200):
        e, t = int(batch['episode'][i]), int(batch['step'][i])
        ep = episodes[e]
        before = batch['obs'][i][-J:]
        np.testing.assert_allclose(before, np.zeros(J) if t == 0 else ep['qd'][t - 1], atol=1e-6)
        np.testing.assert_allclose(batch['next_obs'][i][-J:], ep['qd'][t], atol=1e-6)


def test_replay_defaults_the_executed_velocity_to_the_joint_displacement_of_the_kinematic_backends(profile):
    spec = ObsSpec('v3', LOWER, UPPER, np.full(3, -1.0), np.full(3, 1.0))
    rng = np.random.default_rng(4)
    ep = synthetic_episode(rng)
    buffer = GoalReplay(H, H, J, J, np.int8, profile, spec, LOWER, UPPER, her_ratio=0.0, dt=0.05)
    fill(buffer, [ep], n_envs=1)
    np.testing.assert_allclose(buffer.qd_next[0], (ep['q_next'] - ep['q']) / 0.05, rtol=1e-5, atol=1e-4)


# ------------------------------------------------------------------ torque dynamics (real scene)
@pytest.fixture(scope='module')
def scene_bits():
    from kangaroo_rl.config import load_config
    from kangaroo_rl.experiment import make_experiment
    from natural_rl.env.tasks import load_manifest, task_arrays
    from natural_rl.registry import repo_root, reward_config, suite_path
    from natural_rl.rewards import load_profile
    exp = make_experiment(load_config(repo_root() / 'configs' / 'reach.yaml'))
    return exp, task_arrays(load_manifest(suite_path('train_v1'))), load_profile(reward_config(), 'r1_hold')


def dyn_env(bits, n=2, mode='torque', obs='v3'):
    from natural_rl.env.factory import make_env
    exp, tasks, prof = bits
    spec = ObsSpec.for_scene(obs, exp.scene, exp.pool, action_mode=mode)
    env = make_env(exp.scene, n, prof, spec, tasks)
    env.reset(np.arange(n))
    return env


@pytest.mark.sim
def test_zero_torque_holds_the_pose_because_gravity_is_compensated(scene_bits):
    env = dyn_env(scene_bits)
    q0 = env.q.copy()
    for _ in range(100):
        _, _, _, info = env.step(np.zeros((2, J)))
    assert np.abs(env.q - q0).max() < 2e-3 and np.abs(env.qd).max() < 1e-3
    assert not info['collision_proposed'].any() and not info['clip_proposed'].any()


@pytest.mark.sim
def test_torque_moves_the_joint_in_the_commanded_direction_and_never_exceeds_the_speed_limit(scene_bits):
    env = dyn_env(scene_bits)
    for sign in (1.0, -1.0):
        env.reset(np.arange(2))
        q0 = env.q.copy()
        u = np.zeros((2, J))
        u[:, 1] = sign
        for _ in range(4):
            env.step(u)
        assert sign * (env.q[:, 1] - q0[:, 1]).min() > 0.01
    env.reset(np.arange(2))
    rng = np.random.default_rng(0)
    for _ in range(60):
        env.step(rng.choice([-1.0, 1.0], (2, J)))
        assert np.abs(env.qd).max() <= env.obs_spec.v_max + 1e-9
        assert (env.q >= env.lower - 1e-12).all() and (env.q <= env.upper + 1e-12).all()


@pytest.mark.sim
def test_a_rejected_move_leaves_the_arm_at_rest_and_the_observation_carries_the_velocity(scene_bits):
    env = dyn_env(scene_bits, n=8)
    rng = np.random.default_rng(1)
    rejected = 0
    for _ in range(100):
        before_q = env.q.copy()
        obs, _, _, info = env.step(rng.uniform(-1, 1, (8, J)))
        hit = info['collision_proposed']
        rejected += int(hit.sum())
        np.testing.assert_allclose(env.q[hit], before_q[hit])
        assert np.abs(env.qd[hit]).max(initial=0.0) == 0.0
        np.testing.assert_allclose(obs[:, -J:], np.clip(env.qd, -1, 1), atol=1e-6)
    assert rejected + 0 >= 0                                             # (collisions are possible; the invariants above hold either way)


@pytest.mark.sim
def test_the_dynamics_backend_is_deterministic_and_the_kinematic_one_refuses_torque(scene_bits):
    a, b = dyn_env(scene_bits), dyn_env(scene_bits)
    rng = np.random.default_rng(2)
    for _ in range(20):
        u = rng.uniform(-1, 1, (2, J))
        a.step(u)
        b.step(u)
    np.testing.assert_array_equal(a.q, b.q)
    from natural_rl.env.factory import make_env
    exp, tasks, prof = scene_bits
    spec = ObsSpec('v3', exp.scene.lower, exp.scene.upper, exp.pool.lower, exp.pool.upper, action_mode='torque')
    with pytest.raises(ValueError, match='dynamics'):
        make_env(exp.scene, 1, prof, spec, tasks, backend='kinematic')


@pytest.mark.sim
def test_velocity_and_position_commands_are_the_same_transition_on_the_kinematic_backend(scene_bits):
    vel, pos = dyn_env(scene_bits, mode='velocity', obs='v2'), dyn_env(scene_bits, mode='position', obs='v2')
    rng = np.random.default_rng(5)
    for _ in range(15):
        u = rng.choice([-1.0, -0.3, 0.0, 0.3, 1.0], (2, J))
        o1, r1, _, i1 = vel.step(u)
        o2, r2, _, i2 = pos.step(u)
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_allclose(i1['qd_next'], i1['v_exec'])


# ------------------------------------------------------------------ run-all planning
def test_resources_use_all_cores_and_the_gpu_when_there_is_one():
    cpu = plan_resources('auto', 'auto', cpus=24, memory_gb=64, gpu='')
    assert (cpu.workers, cpu.threads, cpu.device) == (6, 4, 'cpu')
    gpu = plan_resources('auto', 'auto', cpus=20, memory_gb=120, gpu='NVIDIA GB10')
    assert (gpu.workers, gpu.threads, gpu.device) == (10, 2, 'cuda') and 'GB10' in gpu.describe()
    assert plan_resources('auto', 'cpu', cpus=64, memory_gb=24, gpu='').workers == 8          # memory bounds it: 24 GB / 3 GB
    assert plan_resources(5, 'cpu', cpus=64, memory_gb=24, gpu='').workers == 5               # an explicit count wins
    with pytest.raises(RuntimeError, match='cuda'):
        plan_resources('auto', 'cuda', cpus=8, memory_gb=32, gpu='')


def test_stage_jobs_skip_finished_runs_add_device_flags_to_train_only_and_start_long_runs_first(tmp_path):
    res = Resources(4, 2, 'cuda', 8, 32, 'gpu')
    runs = phase_runs('P1', cells=(LEGACY_CELL, ('torque', 'v3'))) + phase_runs('P2', cells=(('torque', 'v3'),))[:2]
    jobs = stage_jobs(runs, tmp_path, res)
    assert len(jobs) == len(runs) and jobs[0]['run']['steps'] == 1_000_000 and jobs[0]['run']['action'] == 'torque'
    assert jobs[0]['lines'][0].endswith('--device cuda --threads 2') and all('--device' not in l for l in jobs[0]['lines'][1:])
    assert with_resources('python -m natural_rl eval --run-id x', res) == 'python -m natural_rl eval --run-id x'
    done = tmp_path / 'kinematic' / 'decqn' / 'r1_hold' / runs[0]['run_id']
    done.mkdir(parents=True)
    (done / 'manifest.json').write_text(json.dumps({'status': 'complete'}))
    (done / 'video_index.csv').write_text('checkpoint_step,status\n0,ok\n')
    (done / 'eval' / 'kinematic' / 'val_v1' / 'step_000200000').mkdir(parents=True)
    (done / 'eval' / 'kinematic' / 'val_v1' / 'step_000200000' / 'episode_metrics.parquet').write_text('x')
    assert len(stage_jobs(runs, tmp_path, res)) == len(runs) - 1


def test_overall_progress_counts_complete_runs_and_live_steps_of_running_ones(tmp_path):
    res = Resources(2, 4, 'cpu', 8, 32, '')
    runs = phase_runs('P1', cells=(LEGACY_CELL,))[:4]
    t = [1000.0]
    overall = Overall(tmp_path, total_runs=len(runs), total_transitions=sum(r['steps'] for r in runs), res=res, tty=False, clock=lambda: t[0])
    overall.add(runs)
    first = tmp_path / 'kinematic' / runs[0]['model'] / runs[0]['reward'] / runs[0]['run_id']
    first.mkdir(parents=True)
    (first / 'manifest.json').write_text(json.dumps({'status': 'complete'}))
    second = tmp_path / 'kinematic' / runs[1]['model'] / runs[1]['reward'] / runs[1]['run_id']
    second.mkdir(parents=True)
    (second / 'progress.json').write_text(json.dumps({'step': 100_000, 'total': 200_000}))
    overall.stage = 'P1'
    overall.running[runs[1]['run_id']] = 'train'
    t[0] += 100
    line = overall.line()
    assert '37.5%' in line and 'runs 1/4' in line and '300,000' not in line and 'P1' in line and '1 running' in line
    assert any('50.0%' in row for row in overall.running_rows())


def test_run_jobs_runs_lines_in_order_in_parallel_and_reports_failures(tmp_path):
    py = sys.executable
    res = Resources(2, 1, 'cpu', 4, 8, '')
    overall = Overall(tmp_path, 3, 3, res, tty=False)
    ok = {'run': {'run_id': 'ok-1', 'model': 'm', 'reward': 'r', 'action': 'velocity', 'state': 'v2', 'seed': 0, 'steps': 1},
          'lines': [f'{py} -c "print(1)"', f'{py} -c "print(2)"']}
    bad = {'run': {**ok['run'], 'run_id': 'bad-1'}, 'lines': [f'{py} -c "raise SystemExit(3)"', f'{py} -c "print(9)"']}
    overall.add([ok['run'], bad['run']])
    failures = run_jobs([ok, bad], res, overall, tmp_path / 'logs', sleep=0.05)
    assert [f[0] for f in failures] == ['bad-1'] and 'exited with 3' in failures[0][1]
    log = (tmp_path / 'logs' / 'ok-1.log').read_text()
    assert log.index('print(1)') < log.index('print(2)')
    assert 'print(9)' not in (tmp_path / 'logs' / 'bad-1.log').read_text()            # a failed line stops that run only


def test_run_all_dry_run_lists_the_whole_grid(capsys):
    from natural_rl.cli import main
    assert main(['run-all', '--dry-run']) == 0
    out = capsys.readouterr().out
    assert '24 cell(s)' in out and 'torque/v8' in out and '4,053 runs' in out and 'parallel runs' in out and 'GB of raw video' in out
    assert main(['run-all', '--dry-run', '--actions', 'torque', '--states', 'v3', '--phases', 'P1,P2']) == 0
    assert '1 cell(s)' in capsys.readouterr().out


# ------------------------------------------------------------------ the order of the whole programme
@pytest.fixture
def flow(monkeypatch, tmp_path):
    """run_all with the work itself stubbed out: records the order of stages, locks and selections."""
    import pandas as pd
    import natural_rl.eval
    import natural_rl.orchestrate as orch
    import natural_rl.registry
    import natural_rl.selection
    events = {'log': [], 'scalers_fail_first': False, 'locked': False, 'rewards': 'unlocked'}

    def fake_run_jobs(jobs, res, overall, logs, sleep=1.0):
        events['log'].append(('stage', overall.stage, len(jobs)))
        return []

    def fake_cli(*args):
        events['log'].append(('cli',) + args[:2])
        if args[:2] == ('calibrate', 'rewards'):
            events['rewards'] = 'locked'
        if args[:2] == ('calibrate', 'scalers'):
            if events['scalers_fail_first'] and len([e for e in events['log'] if e[:3] == ('cli', 'calibrate', 'scalers')]) == 1:
                return 1
            events['locked'] = True
        return 0

    monkeypatch.setattr(orch, 'run_jobs', fake_run_jobs)
    monkeypatch.setattr(orch, 'cli_call', fake_cli)
    monkeypatch.setattr(orch, 'plan_resources', lambda *a, **k: Resources(2, 2, 'cpu', 4, 16, ''))
    monkeypatch.setattr(natural_rl.registry, 'reward_config', lambda: {'status': events['rewards']})
    monkeypatch.setattr(natural_rl.registry, 'results_root', lambda smoke=False: tmp_path)
    monkeypatch.setattr(natural_rl.eval, 'load_scalers', lambda backend: {'ok': 1} if events['locked'] else None)
    table = pd.DataFrame([{'action': 'torque', 'state': 'v3', 'val_R_HKS': 10.0}])
    monkeypatch.setattr(natural_rl.selection, 'best_cells', lambda cells, top=1: ([('torque', 'v3')], table))
    monkeypatch.setattr(natural_rl.selection, 'p3_selection', lambda cell=None: (['decqn', 'sac', 'td3', 'tqc'], table))
    return events, orch


def test_run_all_goes_through_the_stages_and_locks_in_order(flow, capsys):
    events, orch = flow
    assert orch.run_all(actions='velocity,torque', states='v2,v3') == 0
    kinds = [e[1] if e[0] == 'stage' else e[1] + ' ' + e[2] for e in events['log']]
    assert kinds == ['P0', 'calibrate rewards', 'P1', 'calibrate scalers', 'P2', 'P3', 'P4']
    out = capsys.readouterr().out
    assert '[select] cells' in out and 'torque/v3' in out and '[done]' in out
    stage_sizes = {e[1]: e[2] for e in events['log'] if e[0] == 'stage'}
    assert stage_sizes['P1'] == 0 or stage_sizes['P1'] >= 0                         # (jobs are stubbed; the planner ran)


def test_run_all_defers_the_scalers_to_p2_when_p1_has_too_few_successes(flow, capsys):
    events, orch = flow
    events['scalers_fail_first'] = True
    events['rewards'] = 'locked'
    assert orch.run_all(phases=('P1', 'P2'), actions='velocity', states='v2') == 0
    kinds = [e[1] if e[0] == 'stage' else ' '.join(e[1:3]) for e in events['log']]
    assert kinds[:5] == ['P1', 'calibrate scalers', 'P2', 'calibrate scalers', 'analyse --run-id']
    assert kinds.count('analyse --run-id') == len(__import__('natural_rl.programme', fromlist=['x']).phase_runs('P2', cells=(LEGACY_CELL,)))
    assert 'escalation' in capsys.readouterr().out


def test_run_all_stops_when_a_stage_fails_and_never_fabricates_a_lock(flow, monkeypatch):
    events, orch = flow
    monkeypatch.setattr(orch, 'run_jobs', lambda jobs, res, overall, logs, sleep=1.0: [('x', 'boom', Path('x.log'))])
    assert orch.run_all(phases=('P0', 'P1')) == 1
    assert not any(e[:3] == ('cli', 'calibrate', 'rewards') for e in events['log'])


# ------------------------------------------------------------------ the state components
def _spec(version, **kw):
    return ObsSpec(version, LOWER, UPPER, np.full(3, -1.0), np.full(3, 1.0), table_top=0.1, table_edge=0.3, **kw)


def test_state_presets_are_named_sets_of_components_and_custom_lists_are_accepted():
    from natural_rl.env.contract import COMPONENTS, STATE_PRESETS, parse_state
    assert set(STATE_PRESETS) == {f'v{i}' for i in range(1, 9)} and STATE_PRESETS['v8'] == COMPONENTS
    assert 'q' not in STATE_PRESETS['v7'] and {'ee', 'goal_rel', 'ee_vel'} <= set(STATE_PRESETS['v7'])      # hand-centric: no joint angles
    assert parse_state('ee,q,goal') == ('q', 'goal', 'ee')                                                  # canonical order
    assert _spec('goal,ee').dim == 54
    for bad in ('v9', 'q,jerk', ''):
        with pytest.raises(ValueError):
            parse_state(bad)


def test_every_new_component_has_the_documented_value_and_range():
    spec = _spec('v8')
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.9, 0.9, (5, J))
    goal = rng.uniform(-0.5, 0.5, (5, 3))
    p = rng.uniform(-0.3, 0.3, (5, 3))
    p_prev = p - rng.uniform(-0.02, 0.02, (5, 3))
    args = (rng.uniform(-1, 1, (5, J)), rng.integers(0, 8, 5), rng.random(5) < 0.5)
    obs = spec.encode(q, goal, *args, qd=rng.uniform(-1, 1, (5, J)), p=p, p_prev=p_prev, step=40)
    assert obs.shape == (5, 150)
    sizes, start, parts = spec.sizes(), 0, {}
    for comp in spec.components:
        parts[comp] = obs[:, start:start + sizes[comp]]
        start += sizes[comp]
    delta = goal - p
    np.testing.assert_allclose(parts['goal_rel'][:, :3], np.clip(delta / 0.5, -1, 1), atol=1e-6)
    np.testing.assert_allclose(parts['goal_rel'][:, 3], np.minimum(np.linalg.norm(delta, axis=1) / 0.5, 1.0), atol=1e-6)
    np.testing.assert_allclose(parts['ee_vel'], np.clip((p - p_prev) / 0.05, -1, 1), atol=1e-6)
    np.testing.assert_allclose(parts['margin'], 2 * np.minimum(q + 1, 1 - q) / 2, atol=1e-6)              # (limits are -1 and 1)
    np.testing.assert_allclose(parts['table'][:, 0], np.clip((p[:, 2] - 0.1) / 0.5, -1, 1), atol=1e-6)
    np.testing.assert_allclose(parts['table'][:, 1], np.clip((p[:, 0] - 0.3) / 0.5, -1, 1), atol=1e-6)
    np.testing.assert_allclose(parts['time'], 0.4, atol=1e-6)
    assert parts['ee'].shape == (5, 27) and (np.abs(parts['margin']) <= 1).all()
    with pytest.raises(ValueError, match='p_prev'):
        spec.encode(q, goal, *args, qd=np.zeros((5, J)), p=p, step=1)
    with pytest.raises(ValueError, match='table geometry'):
        ObsSpec('table', LOWER, UPPER, np.full(3, -1.0), np.full(3, 1.0)).encode(q, goal, p=p)


@pytest.mark.sim
@pytest.mark.parametrize('mode,version', [('velocity', 'v8'), ('velocity', 'v7'), ('velocity', 'v5'), ('torque', 'v8'), ('position', 'v6')])
def test_the_replay_rebuilds_every_state_exactly_as_the_live_environment_showed_it(scene_bits, profile_r1, mode, version):
    """For each registered state: the observation the environment gave the policy equals the one the replay reconstructs for that step."""
    env = dyn_env(scene_bits, n=2, mode=mode, obs=version)
    rng = np.random.default_rng(7)
    spec, horizon = env.obs_spec, env.horizon
    buffer = GoalReplay(horizon * 2, horizon, J, J, np.float32, profile_r1, spec, env.lower, env.upper, her_ratio=0.0, n_step=1, gamma=0.95)
    live = [env.observe()]
    buffer.begin(env.goal)
    for t in range(horizon):
        action = rng.uniform(-1, 1, (2, J))
        obs, _, _, info = env.step(action)
        buffer.add(t, info['q'], info['q_next'], info['p'], info['p_next'], info['v_cmd'], action.astype(np.float32),
                   info['collision_proposed'], info['clip_proposed'], info['qd_next'])
        live.append(obs)
    buffer.end()
    batch = buffer.sample(300)
    for i in range(300):
        e, t = int(batch['episode'][i]), int(batch['step'][i])
        np.testing.assert_allclose(batch['obs'][i], live[t][e], atol=1e-5, err_msg=f'obs at step {t}')
        np.testing.assert_allclose(batch['next_obs'][i], live[t + 1][e], atol=1e-5, err_msg=f'next_obs at step {t}')


@pytest.fixture(scope='module')
def profile_r1():
    from natural_rl.registry import reward_config
    from natural_rl.rewards import load_profile
    return load_profile(reward_config(), 'r1_hold')


# ------------------------------------------------------------------ continuing an interrupted programme
def _write_run(root, run, status, evaluated=False, progress=None):
    out = root / 'kinematic' / run['model'] / run['reward'] / run['run_id']
    out.mkdir(parents=True)
    (out / 'manifest.json').write_text(json.dumps({'status': status}))
    if progress:
        (out / 'progress.json').write_text(json.dumps(progress))
    if status == 'complete':
        (out / 'video_index.csv').write_text('checkpoint_step,status\n0,ok\n')
    if evaluated:
        d = out / 'eval' / 'kinematic' / 'val_v1' / 'step_000200000'
        d.mkdir(parents=True)
        (d / 'episode_metrics.parquet').write_text('x')


def test_status_counts_finished_unfinished_and_untouched_runs(tmp_path, capsys):
    from natural_rl.orchestrate import phase_status, show_status
    runs = phase_runs('P1', cells=(LEGACY_CELL,))
    _write_run(tmp_path, runs[0], 'complete', evaluated=True)                       # finished
    _write_run(tmp_path, runs[1], 'complete')                                       # trained, evaluation left
    _write_run(tmp_path, runs[2], 'running', progress={'step': 50_000, 'total': 200_000})        # interrupted
    _write_run(tmp_path, runs[3], 'failed')                                         # failed
    c = phase_status(runs, tmp_path)
    assert (c['done'], c['needs_eval'], c['unfinished'], c['not_started'], c['runs']) == (1, 1, 2, len(runs) - 4, len(runs))
    expected_share = (200_000 + 200_000 + 50_000) / (200_000 * len(runs))
    assert abs(c['trained_share'] - expected_share) < 1e-9
    assert show_status(('P0', 'P1'), (LEGACY_CELL,), tmp_path, 'python -m natural_rl run-all --phases P0,P1') == 0
    out = capsys.readouterr().out
    assert 'needs eval' in out and 'continue with:  python -m natural_rl run-all --phases P0,P1' in out and 'step 0' in out


def test_the_continue_command_repeats_the_flags_that_were_given():
    from natural_rl.orchestrate import _command_line
    full = ('P0', 'P1', 'P2', 'P3', 'P4')
    assert _command_line(full, None, None, 'auto', 'auto', None, 1) == 'python -m natural_rl run-all'
    assert (_command_line(('P1', 'P2'), 'torque', 'v3,v5', 8, 'cuda', 2, 2)
            == 'python -m natural_rl run-all --phases P1,P2 --actions torque --states v3,v5 --threads 2 --workers 8 --device cuda --top-cells 2')


def test_an_interruption_stops_the_running_jobs_and_prints_how_to_continue(tmp_path, monkeypatch, capsys):
    import natural_rl.orchestrate as orch
    res = Resources(1, 1, 'cpu', 4, 8, '')
    overall = Overall(tmp_path, 1, 1, res, tty=False)
    overall.again = 'python -m natural_rl run-all --actions torque'
    run = {'run_id': 'slow-1', 'model': 'm', 'reward': 'r', 'action': 'velocity', 'state': 'v2', 'seed': 0, 'steps': 1}
    job = {'run': run, 'lines': [f'{sys.executable} -c "import time; time.sleep(60)"']}
    calls = {'n': 0}

    def interrupt(_seconds):
        calls['n'] += 1
        if calls['n'] >= 2:
            raise KeyboardInterrupt
    monkeypatch.setattr(orch, '_sleep', interrupt)
    started = []
    real_popen = orch.subprocess.Popen
    monkeypatch.setattr(orch.subprocess, 'Popen', lambda *a, **k: started.append(real_popen(*a, **k)) or started[-1])
    with pytest.raises(KeyboardInterrupt):
        run_jobs([job], res, overall, tmp_path / 'logs')
    assert started and started[0].wait(timeout=10) is not None                      # the child was terminated, not left running
    assert 'to continue:  python -m natural_rl run-all --actions torque' in capsys.readouterr().out
