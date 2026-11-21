"""Lossless raw checkpoint video: FFV1 encoding, bit-exact readback verification and a labelled preview.

`raw.mkv` is the record: FFV1 (lossless, intra-only) from the renderer's original RGB frames, 8-bit packed BGR (`bgr0`; this FFmpeg's FFV1 encoder has no 8-bit `gbrp`) so
the RGB<->bgr0 conversion is a pure reordering of channels plus one padding byte. Nothing is overlaid, cropped, retimed, interpolated or colour
graded, and stationary or collision-rejected frames are kept. Verification decodes the file and compares every
frame's SHA256 with the hash of the source frame, with PyAV's bundled FFmpeg and, independently, with the system
`ffmpeg`. `preview.mp4` is a lossy viewing copy, always named as such, and never replaces the record.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

CODEC = 'ffv1'
PIX_FMT = 'bgr0'


def frame_hash(frame) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame, dtype=np.uint8).tobytes()).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_ffv1(frames, path, fps: int = 20) -> dict:
    """Encode uint8 RGB frames (N, H, W, 3) losslessly; returns the encoding record (hashes of every source frame)."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f'frames must be uint8 (N, H, W, 3); got {frames.dtype} {frames.shape}')
    n, height, width, _ = frames.shape
    if n == 0:
        raise ValueError('no frames to encode')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.mkv.partial')
    container = av.open(str(temporary), mode='w', format='matroska')
    stream = container.add_stream(CODEC, rate=fps)
    stream.width, stream.height, stream.pix_fmt = width, height, PIX_FMT
    stream.codec_context.options = {'g': '1', 'slicecrc': '1', 'level': '3'}    # every frame intra-coded and checksummed
    stream.codec_context.time_base = Fraction(1, fps)
    hashes = []
    for k in range(n):
        frame = av.VideoFrame.from_ndarray(frames[k], format='rgb24').reformat(format=PIX_FMT)
        frame.pts = k
        frame.time_base = Fraction(1, fps)
        for packet in stream.encode(frame):
            container.mux(packet)
        hashes.append(frame_hash(frames[k]))
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    temporary.replace(path)                                    # atomic: a crash never leaves a half-written raw.mkv
    return {'codec': CODEC, 'pixel_format': PIX_FMT, 'width': width, 'height': height, 'fps': fps, 'frame_count': n,
            'frame_sha256': hashes, 'video_sha256': file_sha256(path), 'bytes': path.stat().st_size,
            'raw_bytes': int(frames.nbytes)}


def decode_frames(path):
    """All frames of the file as uint8 RGB with their presentation times (s), by PyAV's bundled FFmpeg."""
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format='rgb24'))
            times.append(float(frame.pts * stream.time_base) if frame.pts is not None else float('nan'))
    return frames, times


def _ffmpeg_decode_hashes(path, width, height):
    """Per-frame SHA256 from the SYSTEM ffmpeg (a second, independent decoder), or None if it is not installed."""
    if shutil.which('ffmpeg') is None:
        return None
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                            capture_output=True, check=False)
    if result.returncode != 0:
        return ['ffmpeg-error: ' + result.stderr.decode(errors='replace')[:200]]
    size = width * height * 3
    data = result.stdout
    return [hashlib.sha256(data[i:i + size]).hexdigest() for i in range(0, len(data), size)]


def verify_ffv1(path, expected_hashes, fps: int = 20, width: int | None = None, height: int | None = None) -> dict:
    """Bit-exact readback check. Returns {'ok': bool, 'problems': [...], ...}; never raises on a corrupt file."""
    problems = []
    path = Path(path)
    if not path.exists():
        return {'ok': False, 'problems': [f'missing file {path}'], 'frame_count': 0}
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            codec, pix = stream.codec_context.name, stream.codec_context.pix_fmt
            w, h = stream.codec_context.width, stream.codec_context.height
        if codec != CODEC:
            problems.append(f'codec {codec!r} is not lossless {CODEC!r}')
        if pix != PIX_FMT:
            problems.append(f'pixel format {pix!r} is not {PIX_FMT!r}')
        if (width and w != width) or (height and h != height):
            problems.append(f'resolution {w}x{h} differs from {width}x{height}')
        frames, times = decode_frames(path)
    except Exception as error:                                  # a corrupt container must be reported, not raised
        return {'ok': False, 'problems': [f'decode failed: {type(error).__name__}: {error}'], 'frame_count': 0}
    if len(frames) != len(expected_hashes):
        problems.append(f'{len(frames)} frames decoded, expected {len(expected_hashes)}')
    for k, (frame, want) in enumerate(zip(frames, expected_hashes)):
        if frame_hash(frame) != want:
            problems.append(f'frame {k} differs from the source frame (PyAV decoder)')
            break
    expected_times = [k / fps for k in range(len(expected_hashes))]
    if len(times) == len(expected_times) and not np.allclose(times, expected_times, atol=1e-6):
        problems.append('frame timestamps are not k / fps')
    independent = _ffmpeg_decode_hashes(path, w, h)
    if independent is not None and independent != list(expected_hashes):
        problems.append('system ffmpeg decodes different pixels than the source frames')
    return {'ok': not problems, 'problems': problems, 'frame_count': len(frames), 'codec': codec, 'pixel_format': pix,
            'independent_decoder_checked': independent is not None, 'timestamps_s': times}


def write_preview(frames, path, fps: int = 20) -> bool:
    """Lossy H.264 viewing copy (`preview.mp4`); returns False if the system ffmpeg is unavailable."""
    if shutil.which('ffmpeg') is None:
        return False
    frames = np.asarray(frames)
    n, h, w, _ = frames.shape
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps),
           '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(path)]
    return subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True).returncode == 0


def write_json(path, data) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temporary.replace(path)
