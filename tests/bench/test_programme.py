"""The training programme reproduces the protocol's totals exactly (arithmetic only)."""
import pytest

from natural_rl.checkpoints import scheduled_steps
from natural_rl.programme import phase_runs, programme_totals, train_command, totals


def test_phase_and_grand_totals_match_the_protocol():
    t = programme_totals()
    assert t['P0'] == {'runs': 3, 'transitions': 3_000_000, 'videos': 63}
    assert t['P1'] == {'runs': 60, 'transitions': 12_000_000, 'videos': 300}
    assert t['P2'] == {'runs': 100, 'transitions': 100_000_000, 'videos': 2100}
    assert t['P3'] == {'runs': 200, 'transitions': 200_000_000, 'videos': 4200}
    assert t['P4'] == {'runs': 10, 'transitions': 10_000_000, 'videos': 210}
    assert t['TOTAL'] == {'runs': 373, 'transitions': 325_000_000, 'videos': 6873}


def test_every_run_id_is_unique_across_the_programme():
    ids = [r['run_id'] for p in ('P0', 'P1', 'P2', 'P4') for r in phase_runs(p)]
    ids += [r['run_id'] for r in phase_runs('P3', ['decqn', 'ppo_md', 'sac', 'td3'])]
    assert len(ids) == len(set(ids)) == 373


def test_checkpoint_schedule_has_step_zero_cadence_and_off_cadence_final():
    assert scheduled_steps(1_000_000)[:3] == [0, 50_000, 100_000] and scheduled_steps(1_000_000)[-1] == 1_000_000
    assert len(scheduled_steps(1_000_000)) == 21 and len(scheduled_steps(200_000)) == 5
    steps = scheduled_steps(120_000)
    assert steps == [0, 50_000, 100_000, 120_000]                                # final saved even though off cadence


def test_p3_requires_decqn_plus_three_selected_policies():
    with pytest.raises(ValueError, match='decqn'):
        phase_runs('P3', None)
    with pytest.raises(ValueError):
        phase_runs('P3', ['sac', 'td3', 'tqc', 'crossq'])                      # DecQN must be first


def test_p4_uses_r7_with_fresh_seeds_and_both_budgets():
    runs = phase_runs('P4')
    assert {r['seed'] for r in runs} == set(range(5, 10)) and {r['reward'] for r in runs} == {'r7_balanced'}
    assert {r['budget'] for r in runs} == {'budget_moderate_v1', 'budget_strict_v1'}
    assert 'sac_pidlag-r7-bstrict-s05' in {r['run_id'] for r in runs}


def test_train_command_matches_the_documented_form():
    run = phase_runs('P2')[0]
    assert train_command(run) == ('python -m natural_rl train --model decqn --reward r1_hold --suite train_v1 --steps 1000000 '
                                  '--seed 0 --run-id decqn-r1-s00 --checkpoint-every 50000 --record-every-checkpoint')
    sac_pid = next(r for r in phase_runs('P2') if r['model'] == 'sac_pidlag')
    assert train_command(sac_pid).endswith('--cost-budget-id budget_moderate_v1')


def test_untracked_source_files_make_a_working_tree_dirty(tmp_path):
    import subprocess
    from natural_rl.provenance import git_state
    run = lambda *a: subprocess.run(['git', '-C', str(tmp_path), *a], check=True, capture_output=True)
    run('init', '-q')
    run('config', 'user.email', 'a@b.c')
    run('config', 'user.name', 'x')
    (tmp_path / '.gitignore').write_text('results/\n')
    (tmp_path / 'a.py').write_text('x = 1\n')
    run('add', '.')
    run('commit', '-qm', 'init')
    assert not git_state(tmp_path)['dirty']
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results' / 'big.bin').write_text('ignored output')
    assert not git_state(tmp_path)['dirty']                                   # ignored outputs never make a run dirty
    (tmp_path / 'b.py').write_text('y = 2\n')                                 # a new, uncommitted source file does
    assert git_state(tmp_path)['dirty']


def test_todo_lists_leave_out_finished_runs_and_steps(tmp_path):
    import json
    from natural_rl.programme import pending_lines, phase_runs, run_lines
    run = phase_runs('P1')[0]
    assert pending_lines(run, tmp_path) == run_lines(run)                          # nothing on disk: every line
    out = tmp_path / 'kinematic' / run['model'] / run['reward'] / run['run_id']
    out.mkdir(parents=True)
    (out / 'manifest.json').write_text(json.dumps({'status': 'running'}))
    assert pending_lines(run, tmp_path) == run_lines(run)                          # unfinished: train stops loudly, nothing guessed
    (out / 'manifest.json').write_text(json.dumps({'status': 'complete'}))
    (out / 'video_index.csv').write_text('checkpoint_step,status\n0,ok\n')
    assert pending_lines(run, tmp_path) == run_lines(run)[1:]                      # trained, not yet evaluated: verify + eval remain
    (out / 'eval' / 'kinematic' / 'val_v1' / 'step_000200000').mkdir(parents=True)
    (out / 'eval' / 'kinematic' / 'val_v1' / 'step_000200000' / 'episode_metrics.parquet').write_text('x')
    assert pending_lines(run, tmp_path) == []                                      # everything exists
