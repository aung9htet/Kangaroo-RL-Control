"""Live progress display for `train`: which model and run is executing, how far it is, speed and time left.

On a terminal it redraws one line in place; when output is a file or pipe (e.g. `docker compose run -T`, `| tee`) it prints a plain
line every 5 % of progress (or every 2 minutes). Messages such as checkpoint and video notices are printed above the bar.
"""
from __future__ import annotations

import shutil
import sys
import time


def fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds < 0 or seconds == float('inf'):
        return '--:--:--'
    s = int(seconds)
    return f'{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}'


class Progress:
    def __init__(self, model: str, run_id: str, reward: str, seed: int, total: int, n_checkpoints: int, stream=None,
                 tty: bool | None = None, bar_width: int = 24, min_interval: float = 0.5, plain_every: float = 0.05,
                 plain_seconds: float = 120.0, clock=time.time):
        self.model, self.run_id, self.reward, self.seed = model, run_id, reward, seed
        self.total, self.n_checkpoints, self.saved = max(1, int(total)), int(n_checkpoints), 0
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty() if tty is None else tty
        self.bar_width, self.min_interval, self.plain_every, self.plain_seconds = bar_width, min_interval, plain_every, plain_seconds
        self.clock, self.start = clock, clock()
        self.phase_text, self.step = '', 0
        self._video_started, self.video_seconds, self.videos = None, 0.0, 0
        self._last_draw, self._last_plain_pct, self._last_plain_time, self._drawn = 0.0, -1.0, self.start, False

    # ------------------------------------------------------------------ text
    def line(self, step: int | None = None) -> str:
        step = self.step if step is None else step
        frac = min(1.0, step / self.total)
        filled = int(round(frac * self.bar_width))
        bar = '█' * filled + '░' * (self.bar_width - filled)
        elapsed = self.clock() - self.start
        busy = self.video_seconds + (self.clock() - self._video_started if self._video_started is not None else 0.0)
        train_time = elapsed - busy                         # speed is measured on training time, not on video rendering
        rate = step / train_time if train_time > 0 and step > 0 else 0.0
        videos_left = max(0, self.n_checkpoints - self.saved)
        per_video = self.video_seconds / self.videos if self.videos else 0.0
        eta = (self.total - step) / rate + per_video * videos_left if rate > 0 else float('inf')
        text = (f'{self.model} | {self.run_id} | {self.reward} seed {self.seed} | {bar} {frac * 100:5.1f}% '
                f'{step:,}/{self.total:,} | ckpt {self.saved}/{self.n_checkpoints} | {rate:,.0f} tr/s | '
                f'{fmt_time(elapsed)} elapsed, ETA {fmt_time(eta)}')
        return text + (f' | {self.phase_text}' if self.phase_text else '')

    # ------------------------------------------------------------------ events
    def update(self, step: int, saved: int | None = None) -> None:
        self.step = step
        if saved is not None:
            self.saved = saved
        now = self.clock()
        if self.tty:
            if now - self._last_draw >= self.min_interval or step >= self.total:
                self._draw()
                self._last_draw = now
        else:
            pct = step / self.total
            if pct - self._last_plain_pct >= self.plain_every or now - self._last_plain_time >= self.plain_seconds:
                self.stream.write('[progress] ' + self.line() + '\n')
                self.stream.flush()
                self._last_plain_pct, self._last_plain_time = pct, now

    def _draw(self) -> None:
        width = shutil.get_terminal_size((120, 20)).columns
        self.stream.write('\r' + self.line()[:width - 1] + '\033[K')
        self.stream.flush()
        self._drawn = True

    def log(self, text: str) -> None:
        """Print a message above the bar (never overwritten by it)."""
        if self.tty and self._drawn:
            self.stream.write('\r\033[K')
        self.stream.write(text + '\n')
        self.stream.flush()
        if self.tty:
            self._draw()

    def phase(self, text: str = '') -> None:
        now = self.clock()
        if text and self._video_started is None:
            self._video_started = now
        elif not text and self._video_started is not None:
            self.video_seconds += now - self._video_started
            self.videos += 1
            self._video_started = None
        self.phase_text = text
        if self.tty:
            self._draw()

    def checkpoint_saved(self) -> None:
        self.saved += 1

    def finish(self, status: str) -> None:
        if self.tty and self._drawn:
            self.stream.write('\r\033[K')
        self.stream.write(f'[run] {self.model} | {self.run_id}: {status} after {fmt_time(self.clock() - self.start)}, '
                          f'{self.step:,}/{self.total:,} transitions, {self.saved}/{self.n_checkpoints} checkpoints\n')
        self.stream.flush()
