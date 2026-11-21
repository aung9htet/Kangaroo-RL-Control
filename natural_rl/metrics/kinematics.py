"""Kinematic descriptors of one reaching movement, computed from executed accepted positions.

Scoring uses the movement segment only: it starts at movement onset and ends at the start of the terminal
in-tolerance run, so a steady hold can never inflate smoothness. Derivatives come from one consistent
seven-sample Savitzky-Golay filter applied to the whole episode before slicing; raw finite differences are
reported as a sensitivity. Failures, zero displacement and short or non-finite signals get explicit quality
flags: they are never given an artificially perfect smoothness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .outcomes import distances, terminal_run_start


@dataclass(frozen=True)
class KinematicsConfig:
    dt: float = 0.05
    tolerance: float = 0.02
    sg_window: int = 7
    sg_polyorder: int = 3
    onset_fraction: float = 0.05
    offset_fraction: float = 0.05
    offset_rule: str = 'last_active'      # primary; 'terminal_run_start' and 'terminal_window' are ED5 sensitivities
    terminal_window: int = 5
    min_segment_samples: int = 10
    min_path_m: float = 0.001
    sparc_cutoff_hz: float = 8.0
    sparc_amplitude_threshold: float = 0.05
    sparc_padding: int = 4
    peak_prominence: float = 0.10
    pause_fraction: float = 0.01
    reversal_threshold_rad: float = 1.0e-4
    template_points: int = 101

    @classmethod
    def from_yaml(cls, cfg: dict) -> 'KinematicsConfig':
        return cls(
            dt=float(cfg['sampling']['dt_s']), tolerance=float(cfg['outcomes']['tolerance_m']),
            sg_window=int(cfg['derivatives']['window']), sg_polyorder=int(cfg['derivatives']['polyorder']),
            onset_fraction=float(cfg['segmentation']['onset_speed_fraction']),
            offset_fraction=float(cfg['segmentation']['offset_speed_fraction']),
            offset_rule=str(cfg['segmentation']['offset_rule']),
            terminal_window=int(cfg['outcomes']['terminal_window']),
            min_segment_samples=int(cfg['segmentation']['min_segment_samples']),
            min_path_m=float(cfg['segmentation']['min_path_m']),
            sparc_cutoff_hz=float(cfg['sparc']['cutoff_hz']),
            sparc_amplitude_threshold=float(cfg['sparc']['amplitude_threshold']),
            sparc_padding=int(cfg['sparc']['padding_exponent']),
            peak_prominence=float(cfg['speed_peaks']['prominence_fraction']),
            pause_fraction=float(cfg['pauses']['speed_fraction']),
            reversal_threshold_rad=float(cfg['joint_reversals']['threshold_rad']),
            template_points=int(cfg['template']['n_points']))


def min_jerk_speed(s):
    """Analytic minimum-jerk speed profile on normalised time s in [0, 1] (metric reference only)."""
    s = np.asarray(s, dtype=np.float64)
    return 30.0 * s ** 2 * (1.0 - s) ** 2


def savgol_derivatives(p, dt, window=7, polyorder=3):
    """Velocity, acceleration and jerk of positions p (N, D) by Savitzky-Golay local polynomial fits."""
    p = np.asarray(p, dtype=np.float64)
    if len(p) < window:
        raise ValueError(f'need at least {window} samples, got {len(p)}')
    return tuple(savgol_filter(p, window, polyorder, deriv=k, delta=dt, axis=0, mode='interp')
                 for k in (1, 2, 3))


def sparc(speed, fs, cutoff_hz=8.0, amplitude_threshold=0.05, padding=4):
    """Spectral arc length of a speed profile (Balasubramanian et al. 2015); less negative = smoother.

    The cutoff must lie below the Nyquist frequency of the sampled signal; at 20 samples/s a 20 Hz cutoff
    would treat unsupported frequencies as data, so it is refused.
    """
    if cutoff_hz >= fs / 2.0:
        raise ValueError(f'SPARC cutoff {cutoff_hz} Hz must be below the Nyquist frequency {fs / 2.0} Hz')
    speed = np.asarray(speed, dtype=np.float64)
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padding))
    freqs = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    peak = mag.max()
    if not np.isfinite(peak) or peak <= 0:
        return float('nan')
    mag = mag / peak
    selected = freqs <= cutoff_hz
    f_sel, m_sel = freqs[selected], mag[selected]
    above = np.flatnonzero(m_sel >= amplitude_threshold)
    if len(above) < 2:
        return float('nan')
    f_sel, m_sel = f_sel[above[0]:above[-1] + 1], m_sel[above[0]:above[-1] + 1]
    span = f_sel[-1] - f_sel[0]
    if span <= 0:
        return float('nan')
    return float(-np.sum(np.sqrt((np.diff(f_sel) / span) ** 2 + np.diff(m_sel) ** 2)))


def template_rmse(speed, n_points=101):
    """RMSE between the segment's speed (normalised to unit mean) and 30 s^2 (1-s)^2 over s in [0, 1]."""
    speed = np.asarray(speed, dtype=np.float64)
    if len(speed) < 2:
        return float('nan')
    s = np.linspace(0.0, 1.0, len(speed))
    grid = np.linspace(0.0, 1.0, n_points)
    resampled = np.interp(grid, s, speed)
    mean = np.trapz(resampled, grid)          # mean speed over normalised time
    if not np.isfinite(mean) or mean <= 0:
        return float('nan')
    return float(np.sqrt(np.mean((resampled / mean - min_jerk_speed(grid)) ** 2)))


def joint_reversal_rate(q, dt, threshold):
    """Executed joint-velocity sign changes per second (summed over joints; sub-threshold motion = zero)."""
    dq = np.diff(np.asarray(q, dtype=np.float64), axis=0)
    dq = np.where(np.abs(dq) < threshold, 0.0, dq)
    count = 0
    for joint in dq.T:
        signs = np.sign(joint[joint != 0.0])
        count += int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
    duration = dt * len(dq)
    return count / duration if duration > 0 else float('nan')


_NAN_FIELDS = ('onset_k', 'offset_k', 'segment_duration_s', 'segment_path_m', 'path_efficiency', 'dimensionless_jerk',
               'log1p_dj', 'dimensionless_jerk_fd', 'sparc', 'template_rmse', 'speed_peaks', 'peaks_excess',
               'pause_fraction', 'joint_reversal_rate', 'peak_speed_m_s')


def _empty(flag):
    return {**{k: float('nan') for k in _NAN_FIELDS}, 'valid': False, 'quality_flag': flag}


def segment_kinematics(ee_xyz, q, goal_xyz, terminal_hold: bool, cfg: KinematicsConfig = KinematicsConfig(),
                       sparc_cutoff_hz: float | None = None) -> dict:
    """Descriptors of one episode.

    ee_xyz: (N, 3) accepted end-effector positions in metres, N = decisions + 1.
    q:      (N, J) accepted joint angles in radians.
    terminal_hold: the terminal 5-step hold flag from `episode_outcomes`.
    Non-terminal-hold episodes are described over onset..end for diagnostics only (`no_terminal_hold`).
    """
    p = np.asarray(ee_xyz, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    goal = np.asarray(goal_xyz, dtype=np.float64)
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        return _empty('nonfinite')
    if len(p) < cfg.sg_window:
        return _empty('short_segment')

    velocity, _, jerk = savgol_derivatives(p, cfg.dt, cfg.sg_window, cfg.sg_polyorder)
    speed = np.linalg.norm(velocity, axis=1)
    peak = float(speed.max())
    if peak <= 0 or not np.isfinite(peak):
        return _empty('zero_displacement')
    onset = int(np.flatnonzero(speed >= cfg.onset_fraction * peak)[0])

    d = distances(p, goal)
    flag = ''
    last = len(p) - 1
    # The steady hold must never inflate smoothness, so a terminal-hold episode is cut before its terminal
    # window; an episode without a terminal hold is described over all of its motion (diagnostics only).
    limit = last - cfg.terminal_window if terminal_hold else last
    if not terminal_hold:
        flag = 'no_terminal_hold'
    if cfg.offset_rule == 'last_active':
        # End of the final sub-movement: the last sample at or above the offset speed. Pauses and a second
        # sub-movement stay inside the segment (they are penalised by the descriptors, not hidden).
        active = np.flatnonzero(speed[:limit + 1] >= cfg.offset_fraction * peak)
        offset = int(active[-1]) if len(active) else onset
    elif cfg.offset_rule == 'terminal_window':
        offset = limit
    elif cfg.offset_rule == 'terminal_run_start':
        start = terminal_run_start(d, cfg.tolerance) if terminal_hold else None
        offset = last if start is None else int(start)
    else:
        raise ValueError(f'unknown offset_rule {cfg.offset_rule!r}')
    offset = max(onset, min(offset, limit))
    if offset - onset + 1 < cfg.min_segment_samples:
        return {**_empty('short_segment'), 'onset_k': float(onset), 'offset_k': float(offset)}

    seg = slice(onset, offset + 1)
    p_seg, v_seg, j_seg, q_seg = p[seg], speed[seg], jerk[seg], q[seg]
    length = float(np.linalg.norm(np.diff(p_seg, axis=0), axis=1).sum())
    if length < cfg.min_path_m:
        return {**_empty('zero_displacement'), 'onset_k': float(onset), 'offset_k': float(offset)}
    duration = (len(p_seg) - 1) * cfg.dt
    direct = float(np.linalg.norm(goal - p[0]))
    # Efficiency uses the executed pre-hold path from the first sample (movement before onset is below 5 % of
    # peak speed, so it is negligible); the steady hold is excluded. It can exceed 1 only by the 2 cm slack.
    pre_path = float(np.linalg.norm(np.diff(p[:offset + 1], axis=0), axis=1).sum())

    dj = (duration ** 5 / length ** 2) * float(np.trapz(np.sum(j_seg ** 2, axis=1), dx=cfg.dt))
    jerk_fd = np.diff(p, n=3, axis=0) / cfg.dt ** 3                       # sample k covers k..k+3
    fd_seg = jerk_fd[max(onset, 0):max(min(offset - 2, len(jerk_fd)), onset)]
    dj_fd = (duration ** 5 / length ** 2) * float(np.sum(np.sum(fd_seg ** 2, axis=1)) * cfg.dt) if len(fd_seg) else float('nan')

    cutoff = cfg.sparc_cutoff_hz if sparc_cutoff_hz is None else sparc_cutoff_hz
    peaks, _ = find_peaks(v_seg, prominence=cfg.peak_prominence * float(v_seg.max()))
    seg_peak = float(v_seg.max())
    result = {
        'onset_k': float(onset), 'offset_k': float(offset),
        'segment_duration_s': float(duration), 'segment_path_m': length,
        'path_efficiency': direct / pre_path if pre_path > 0 else float('nan'),
        'dimensionless_jerk': dj, 'log1p_dj': float(np.log1p(dj)), 'dimensionless_jerk_fd': dj_fd,
        'sparc': sparc(v_seg, 1.0 / cfg.dt, cutoff, cfg.sparc_amplitude_threshold, cfg.sparc_padding),
        'template_rmse': template_rmse(v_seg, cfg.template_points),
        'speed_peaks': float(len(peaks)), 'peaks_excess': float(max(len(peaks) - 1, 0)),
        'pause_fraction': float(np.mean(v_seg < cfg.pause_fraction * seg_peak)),
        'joint_reversal_rate': joint_reversal_rate(q_seg, cfg.dt, cfg.reversal_threshold_rad),
        'peak_speed_m_s': seg_peak,
    }
    finite = all(np.isfinite(result[k]) for k in ('path_efficiency', 'log1p_dj', 'sparc', 'template_rmse',
                                                  'pause_fraction', 'joint_reversal_rate'))
    if not finite:
        flag = flag or 'nonfinite'
    return {**result, 'valid': bool(finite and not flag), 'quality_flag': flag}
