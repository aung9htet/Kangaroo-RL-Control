"""End-to-end integration on the real scene: train with checkpoint videos, verify, replay, evaluate, corrupt, re-verify.

Slow (about two minutes: it renders and encodes real raw videos). Run inside the container.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from natural_rl.cli import main

pytestmark = [pytest.mark.sim, pytest.mark.slow]
RUN = 'it-decqn'


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp('results')
    import os
    os.environ['NATURAL_RL_RESULTS'] = str(root)
    try:
        status = main(['train', '--model', 'decqn', '--reward', 'r1_hold', '--suite', 'train_v1', '--steps', '1600', '--seed', '0',
                       '--run-id', RUN, '--checkpoint-every', '1600', '--record-every-checkpoint', '--smoke',
                       '--set', 'hyper.warmup=800', 'hyper.replay=16000'])
        assert status == 0
        yield next((root / 'smoke' / 'runs').glob(f'*/*/*/{RUN}'))
    finally:
        os.environ.pop('NATURAL_RL_RESULTS', None)


def test_run_is_complete_with_a_verified_video_for_every_checkpoint(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['smoke'] and manifest['obs_version'] == 'v2'
    index = pd.read_csv(run / 'checkpoint_index.csv')
    videos = pd.read_csv(run / 'video_index.csv')
    assert list(index['checkpoint_step']) == [0, 1600] == list(videos['checkpoint_step'])      # step 0 + final, one video each
    assert set(videos['status']) == {'ok'} and videos['verified'].all() and (videos['frame_count'] == 101).all()
    for step in (0, 1600):
        d = run / 'videos' / f'step_{step:09d}' / 'val_001'
        for name in ('raw.mkv', 'preview.mp4', 'video_manifest.json', 'rollout.parquet'):
            assert (d / name).exists(), name
        vm = json.loads((d / 'video_manifest.json').read_text())
        assert vm['codec'] == 'ffv1' and vm['width'] == 640 and vm['height'] == 480 and vm['fps'] == 20 and vm['decision_count'] == 100
        assert vm['camera_id'] == 'canonical_v1' and vm['task_id'] == 'val_001' and vm['evaluation_seed'] == 20261121
    assert main(['video', 'verify', '--run-id', RUN, '--all-checkpoints']) == 0


def test_rerecording_a_saved_checkpoint_reproduces_identical_pixels_without_overwriting(run):
    before = (run / 'videos' / 'step_000001600' / 'val_001' / 'raw.mkv').stat().st_mtime_ns
    assert main(['video', 'record', '--run-id', RUN, '--checkpoint-step', '1600', '--suite', 'checkpoint_video_v1']) == 0
    assert (run / 'videos' / 'step_000001600' / 'val_001' / 'raw.mkv').stat().st_mtime_ns == before
    assert main(['video', 'backfill', '--run-id', RUN, '--suite', 'checkpoint_video_v1']) == 0     # nothing missing: no rewrite


def test_test_checkpoint_is_never_chosen_without_validation_results_then_evaluation_writes_all_tables(run):
    with pytest.raises(FileNotFoundError, match='validation evaluation'):
        main(['eval', '--run-id', RUN, '--suite', 'test_v1', '--manifest', 'configs/eval/test_v1.json'])
    assert main(['eval', '--run-id', RUN, '--suite', 'val_v1', '--all-checkpoints']) == 0
    assert main(['eval', '--run-id', RUN, '--suite', 'test_v1', '--manifest', 'configs/eval/test_v1.json']) == 0
    selected = sorted((run / 'eval' / 'kinematic' / 'test_v1').glob('step_*'))
    assert len(selected) == 1                                                     # one validation-selected checkpoint, not all
    out = selected[0]
    metrics = pd.read_parquet(out / 'episode_metrics.parquet')
    assert len(metrics) == 200 and metrics['task_id'].is_unique and (metrics['split'] == 'test_v1').all()
    for column in ('terminal_hold', 'first_entry_s', 'final_error_m', 'path_m', 'duration_s', 'HKS', 'R_HKS', 'quality_flag', 'seed',
                   'checkpoint_step', 'backend'):
        assert column in metrics.columns, column
    assert (metrics['hks_status'] == 'scalers_not_locked').all()                    # never a guessed score before the lock
    assert len(pd.read_parquet(out / 'trajectories.parquet')) == 200 * 101
    assert len(pd.read_parquet(out / 'action_events.parquet')) == 200 * 100
    assert main(['analyse', '--run-id', RUN]) == 0


def test_a_manifest_other_than_the_registered_one_is_refused(run):
    assert main(['eval', '--run-id', RUN, '--suite', 'test_v1', '--manifest', 'configs/eval/val_v1.json']) == 2


def test_physics_evaluation_is_a_separate_stratum_with_its_own_tables(run):
    from natural_rl.cli import _run_context
    from natural_rl.eval import evaluate_checkpoint
    out, manifest, scene, spec, profile = _run_context(RUN, keep_env=True)
    metrics = evaluate_checkpoint(out, manifest, scene, spec, profile, 0, 'val_v1', backend='physics', max_tasks=2)
    assert len(metrics) == 2 and (metrics['backend'] == 'physics').all()
    physics = run / 'eval' / 'physics' / 'val_v1' / 'step_000000000'
    kinematic = run / 'eval' / 'kinematic' / 'val_v1' / 'step_000000000'
    assert (physics / 'episode_metrics.parquet').exists() and (kinematic / 'episode_metrics.parquet').exists()
    assert len(pd.read_parquet(kinematic / 'episode_metrics.parquet')) == 100             # the fast-backend table is untouched
    frames = pd.read_parquet(physics / 'trajectories.parquet')
    assert (frames['backend'] == 'physics').all() and len(frames) == 2 * 101


def test_verification_fails_loudly_on_a_corrupted_raw_video(run):
    raw = run / 'videos' / 'step_000000000' / 'val_001' / 'raw.mkv'
    data = bytearray(raw.read_bytes())
    for i in range(len(data) // 2, len(data) // 2 + 300):
        data[i] ^= 0xFF
    raw.write_bytes(bytes(data))
    assert main(['video', 'verify', '--run-id', RUN, '--all-checkpoints']) == 1


def test_trajectory_export_is_flat_with_units_in_the_column_names(run):
    assert main(['export', '--run-id', RUN, '--kind', 'trajectories', '--suite', 'val_v1']) == 0
    files = sorted((run / 'exports' / 'trajectories').glob('kinematic_val_v1_step_*.csv'))
    assert files, 'no exported csv'
    frame = pd.read_csv(files[0])
    for column in ('task_id', 't_s', 'q1_rad', 'q7_rad', 'ee_x_m', 'ee_z_m', 'goal_y_m', 'vx_m_s', 'coordinate_frame', 'checkpoint_step'):
        assert column in frame.columns, column
    assert len(frame) == 100 * 101 and frame['task_id'].nunique() == 100
    assert main(['export', '--run-id', RUN, '--kind', 'geometry', '--suite', 'val_v1']) == 2        # not implemented: says so
