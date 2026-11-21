"""Lossless raw-video layer: bit-exact roundtrip, honest verification (detects corruption), no padding."""
import shutil

import numpy as np
import pytest

from natural_rl.video import decode_frames, file_sha256, frame_hash, verify_ffv1, write_ffv1, write_preview

W, H = 64, 48


def make_frames(n, seed=0, still_tail=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    frames = np.stack([np.roll(base, k * 2, axis=1) ^ rng.integers(0, 4, (H, W, 3), dtype=np.uint8) for k in range(n)])
    if still_tail:
        frames[n - still_tail:] = frames[n - still_tail - 1]                  # stationary frames must be kept as they are
    return frames


def test_roundtrip_is_bit_exact_for_every_frame(tmp_path):
    frames = make_frames(101, still_tail=12)
    record = write_ffv1(frames, tmp_path / 'raw.mkv')
    decoded, times = decode_frames(tmp_path / 'raw.mkv')
    assert len(decoded) == 101 == record['frame_count']
    assert all(np.array_equal(a, b) for a, b in zip(frames, decoded))
    np.testing.assert_allclose(times, np.arange(101) / 20.0, atol=1e-9)
    assert record['codec'] == 'ffv1' and record['pixel_format'] == 'bgr0'
    assert record['video_sha256'] == file_sha256(tmp_path / 'raw.mkv')


def test_stationary_frames_are_not_dropped_and_early_end_is_not_padded(tmp_path):
    frames = make_frames(37, still_tail=10)
    write_ffv1(frames, tmp_path / 'raw.mkv')
    decoded, _ = decode_frames(tmp_path / 'raw.mkv')
    assert len(decoded) == 37                                                  # not 101: an early end is logged, not padded


def test_verify_passes_a_good_file_with_both_decoders(tmp_path):
    frames = make_frames(21)
    record = write_ffv1(frames, tmp_path / 'raw.mkv')
    out = verify_ffv1(tmp_path / 'raw.mkv', record['frame_sha256'], width=W, height=H)
    assert out['ok'], out['problems']
    if shutil.which('ffmpeg'):
        assert out['independent_decoder_checked']


def test_verify_detects_a_different_source_frame(tmp_path):
    frames = make_frames(21)
    record = write_ffv1(frames, tmp_path / 'raw.mkv')
    wrong = list(record['frame_sha256'])
    wrong[7] = frame_hash(np.zeros((H, W, 3), dtype=np.uint8))
    out = verify_ffv1(tmp_path / 'raw.mkv', wrong)
    assert not out['ok'] and any('frame 7' in p for p in out['problems'])


def test_verify_detects_a_missing_frame_and_a_missing_file(tmp_path):
    frames = make_frames(21)
    record = write_ffv1(frames[:20], tmp_path / 'raw.mkv')                       # one frame short of what the run produced
    out = verify_ffv1(tmp_path / 'raw.mkv', [frame_hash(f) for f in frames])
    assert not out['ok'] and any('20 frames decoded, expected 21' in p for p in out['problems'])
    assert not verify_ffv1(tmp_path / 'nope.mkv', record['frame_sha256'])['ok']


def test_verify_detects_file_corruption_without_raising(tmp_path):
    frames = make_frames(21)
    record = write_ffv1(frames, tmp_path / 'raw.mkv')
    data = bytearray((tmp_path / 'raw.mkv').read_bytes())
    for offset in range(len(data) // 3, len(data) // 3 + 200):                    # damage the middle of the stream
        data[offset] ^= 0xFF
    (tmp_path / 'raw.mkv').write_bytes(bytes(data))
    out = verify_ffv1(tmp_path / 'raw.mkv', record['frame_sha256'])
    assert not out['ok'] and out['problems']


def test_a_lossy_file_is_rejected_as_not_the_record(tmp_path):
    if not shutil.which('ffmpeg'):
        pytest.skip('system ffmpeg not installed')
    frames = make_frames(21)
    record = write_ffv1(frames, tmp_path / 'raw.mkv')
    assert write_preview(frames, tmp_path / 'preview.mp4')
    out = verify_ffv1(tmp_path / 'preview.mp4', record['frame_sha256'])          # an mp4 can never pass as the raw record
    assert not out['ok'] and any('lossless' in p or 'pixel format' in p for p in out['problems'])


def test_writer_refuses_wrong_input_and_leaves_no_partial_file(tmp_path):
    with pytest.raises(ValueError):
        write_ffv1(np.zeros((3, H, W, 3), dtype=np.float32), tmp_path / 'raw.mkv')
    with pytest.raises(ValueError):
        write_ffv1(np.zeros((0, H, W, 3), dtype=np.uint8), tmp_path / 'raw.mkv')
    assert not list(tmp_path.glob('*'))


def test_compression_ratio_is_reported_for_storage_planning(tmp_path):
    record = write_ffv1(make_frames(30), tmp_path / 'raw.mkv')
    assert record['raw_bytes'] == 30 * H * W * 3 and 0 < record['bytes']
