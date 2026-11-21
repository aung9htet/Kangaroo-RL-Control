"""`run-all`: the whole programme (every phase, action mode and state) from one command, on all the resources of the machine.

It plans the registered stages (P0, reward lock, P1, scaler lock, P2, cell and controller selection, P3, P4), turns every run into
its ordered command lines (`programme.pending_lines`, so finished work is skipped), and runs many runs at once: one process per
run, as many as the CPU cores and memory allow, all sharing the GPU when there is one. One overall display shows the progress of
the whole programme. The calibration locks and the P3/P4 selections happen between the stages, from real results only.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import programme
from .progress import fmt_time


# ------------------------------------------------------------------ resources
@dataclass(frozen=True)
class Resources:
    workers: int
    threads: int
    device: str
    cpus: int
    memory_gb: float
    gpu: str

    def describe(self) -> str:
        return (f'{self.workers} parallel runs x {self.threads} thread(s), device {self.device}'
                f'{" (" + self.gpu + ")" if self.gpu else ""}, {self.cpus} CPU cores, {self.memory_gb:.0f} GB memory available')


def detect_gpu() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
    except Exception:                                                  # no torch / no driver: CPU only
        return ''


def available_memory_gb() -> float:
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 16.0


def plan_resources(workers='auto', device='auto', threads=None, cpus=None, memory_gb=None, gpu=None, gb_per_run: float = 3.0) -> Resources:
    """Use what the machine has: one run per `threads` cores (2 with a GPU, which does the learning; 4 on CPU), bounded by memory."""
    cpus = cpus or os.cpu_count() or 1
    memory_gb = available_memory_gb() if memory_gb is None else memory_gb
    gpu = detect_gpu() if gpu is None else gpu
    if device == 'auto':
        device = 'cuda' if gpu else 'cpu'
    if device == 'cuda' and not gpu:
        raise RuntimeError('--device cuda was requested but no CUDA device is visible (see README, GPU)')
    threads = int(threads) if threads else (2 if device == 'cuda' else 4)
    auto = max(1, min(cpus // threads, int(memory_gb // gb_per_run)))
    n = auto if workers in (None, 'auto') else int(workers)
    return Resources(max(1, n), threads, device, cpus, memory_gb, gpu if device == 'cuda' else '')


# ------------------------------------------------------------------ planning
def run_dir_of(run_id: str, runs_root: Path):
    found = list(Path(runs_root).glob(f'*/*/*/{run_id}'))
    return found[0] if len(found) == 1 else None


def run_fraction(run: dict, runs_root: Path) -> float:
    """Share of a run's training that is done: 1 if complete, the live step share if running, else 0."""
    out = run_dir_of(run['run_id'], runs_root)
    if out is None:
        return 0.0
    try:
        if json.loads((out / 'manifest.json').read_text()).get('status') == 'complete':
            return 1.0
    except (OSError, ValueError):
        pass
    try:
        info = json.loads((out / 'progress.json').read_text())
        return min(1.0, info['step'] / max(1, info['total']))
    except (OSError, ValueError, KeyError):
        return 0.0


def with_resources(line: str, res: Resources) -> str:
    """Add the device and thread count to a `train` line (other lines are unchanged)."""
    if ' natural_rl train ' not in line:
        return line
    return f'{line} --device {res.device} --threads {res.threads}'


def stage_jobs(runs: list, runs_root: Path, res: Resources, with_eval: bool = True) -> list:
    jobs = []
    for run in runs:
        lines = programme.pending_lines(run, runs_root, with_eval)
        if lines:
            jobs.append({'run': run, 'lines': [with_resources(l, res) for l in lines]})
    # long runs first and the slow dynamics runs first, so the tail of a stage is short
    jobs.sort(key=lambda j: (-j['run']['steps'], j['run']['action'] != 'torque', j['run']['run_id']))
    return jobs


# ------------------------------------------------------------------ overall display
class Overall:
    def __init__(self, runs_root: Path, total_runs: int, total_transitions: int, res: Resources, stream=None, tty=None, clock=time.time):
        self.root, self.total_runs, self.total, self.res = Path(runs_root), total_runs, total_transitions, res
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty() if tty is None else tty
        self.clock, self.start = clock, clock()
        self.known: dict = {}            # run_id -> run dict of every run planned so far
        self.stage, self.running, self.done_at_start, self._drawn_rows, self._last_plain = '', {}, None, 0, 0.0

    def add(self, runs: list) -> None:
        for run in runs:
            self.known[run['run_id']] = run
        if self.done_at_start is None:
            self.done_at_start = self.done_transitions()

    def done_transitions(self) -> float:
        return sum(r['steps'] * run_fraction(r, self.root) for r in self.known.values())

    def done_runs(self) -> int:
        return sum(1 for r in self.known.values() if run_fraction(r, self.root) >= 1.0)

    def line(self) -> str:
        done = self.done_transitions()
        frac = min(1.0, done / max(1, self.total))
        filled = int(round(frac * 28))
        elapsed = self.clock() - self.start
        gained = done - (self.done_at_start or 0.0)
        rate = gained / elapsed if elapsed > 5 and gained > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else float('inf')
        return (f'[overall] {"█" * filled}{"░" * (28 - filled)} {frac * 100:5.1f}% | {self.stage or "starting"} | '
                f'runs {self.done_runs():,}/{self.total_runs:,} | {done / 1e6:,.1f}M/{self.total / 1e6:,.0f}M transitions | '
                f'{len(self.running)} running | {rate:,.0f} tr/s | {fmt_time(elapsed)} elapsed, ETA {fmt_time(eta)}')

    def running_rows(self, limit: int = 8) -> list:
        rows = []
        for run_id, text in sorted(self.running.items())[:limit]:
            run = self.known.get(run_id)
            pct = 100 * run_fraction(run, self.root) if run else 0
            rows.append(f'  {run["model"] if run else "?":10s} {run["reward"].split("_")[0] if run else "":4s} '
                        f'{(run["action"] + "/" + run["state"]) if run else "":14s} s{run["seed"] if run else 0:<2d} {pct:5.1f}%  {text}')
        if len(self.running) > limit:
            rows.append(f'  ... and {len(self.running) - limit} more running')
        return rows

    def draw(self, force: bool = False) -> None:
        now = self.clock()
        if self.tty:
            width = os.get_terminal_size().columns if self.stream.isatty() else 140
            rows = [self.line()] + self.running_rows()
            if self._drawn_rows:
                self.stream.write(f'\033[{self._drawn_rows}F')
            self.stream.write(''.join(r[:width - 1] + '\033[K\n' for r in rows))
            self.stream.write('\033[J')
            self.stream.flush()
            self._drawn_rows = len(rows)
        elif force or now - self._last_plain >= 60:
            self.stream.write(self.line() + '\n')
            self.stream.flush()
            self._last_plain = now

    def message(self, text: str) -> None:
        if self.tty and self._drawn_rows:
            self.stream.write(f'\033[{self._drawn_rows}F\033[J')
            self._drawn_rows = 0
        self.stream.write(text + '\n')
        self.stream.flush()


# ------------------------------------------------------------------ execution
_sleep = time.sleep                      # one name for the polling pause, so tests can interrupt the loop without touching time.sleep


@dataclass
class Job:
    run: dict
    lines: list
    log: Path
    index: int = 0
    proc: object = None
    failed: str = ''
    handle: object = field(default=None, repr=False)

    def start_next(self) -> None:
        line = self.lines[self.index]
        self.handle = open(self.log, 'ab')
        self.handle.write(f'\n>>> {line}\n'.encode())
        self.handle.flush()
        self.proc = subprocess.Popen(shlex.split(line), stdout=self.handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=Path(__file__).resolve().parents[1])

    def poll(self) -> bool:
        """Advance the job; True when it has finished (successfully or not)."""
        if self.proc is None:
            self.start_next()
            return False
        code = self.proc.poll()
        if code is None:
            return False
        self.handle.close()
        if code != 0:
            self.failed = f'line {self.index + 1} exited with {code}: {self.lines[self.index]}'
            return True
        self.index += 1
        if self.index >= len(self.lines):
            return True
        self.start_next()
        return False

    def label(self) -> str:
        line = self.lines[min(self.index, len(self.lines) - 1)].split()
        verb = line[3] if len(line) > 3 else ''
        return f'{verb} {" ".join(line[4:6]) if verb in ("eval", "video", "analyse") else ""}'.strip()


def run_jobs(jobs: list, res: Resources, overall: Overall, logs: Path, sleep: float = 1.0) -> list:
    """Run the jobs, at most `res.workers` at a time. Returns the failures as (run_id, reason, log path)."""
    logs.mkdir(parents=True, exist_ok=True)
    queue = [Job(j['run'], j['lines'], logs / f'{j["run"]["run_id"]}.log') for j in jobs]
    active, failures = [], []
    try:
        while queue or active:
            while queue and len(active) < res.workers:
                job = queue.pop(0)
                active.append(job)
                overall.running[job.run['run_id']] = ''
            for job in list(active):
                finished = job.poll()
                overall.running[job.run['run_id']] = job.label()
                if finished:
                    active.remove(job)
                    overall.running.pop(job.run['run_id'], None)
                    if job.failed:
                        failures.append((job.run['run_id'], job.failed, job.log))
                        overall.message(f'[FAILED] {job.run["run_id"]}: {job.failed}\n         log: {job.log}')
            overall.draw()
            _sleep(sleep)
    except KeyboardInterrupt:
        for job in active:
            job.proc.terminate()
        overall.message('[interrupted] the running jobs were stopped. Finished runs are kept; the interrupted ones start again from step 0.\n'
                        f'            to continue:  {getattr(overall, "again", "python -m natural_rl run-all")}')
        raise
    return failures


# ------------------------------------------------------------------ the programme
def python_cli(*args) -> list:
    return [sys.executable, '-m', 'natural_rl', *args]


def cli_call(*args) -> int:
    return subprocess.call(python_cli(*args), cwd=Path(__file__).resolve().parents[1])


def phase_status(runs: list, runs_root: Path) -> dict:
    """Counts of a phase's runs by state: done, needs_eval (trained; verification or evaluation left), unfinished (interrupted or
    failed: it will be replaced and trained again from step 0), not_started; plus the share of its transitions already trained."""
    counts = {'done': 0, 'needs_eval': 0, 'unfinished': 0, 'not_started': 0}
    for run in runs:
        out = run_dir_of(run['run_id'], runs_root)
        if out is None or not (out / 'manifest.json').exists():
            counts['not_started'] += 1
        elif programme.pending_lines(run, runs_root) == []:
            counts['done'] += 1
        elif run_fraction(run, runs_root) >= 1.0:
            counts['needs_eval'] += 1
        else:
            counts['unfinished'] += 1
    total = sum(r['steps'] for r in runs)
    counts['trained_share'] = sum(r['steps'] * run_fraction(r, runs_root) for r in runs) / total if total else 1.0
    counts['runs'] = len(runs)
    return counts


def show_status(phases, cells, runs_root: Path, command: str) -> int:
    """What is finished and what is left of the programme, read from results/ only (nothing is started)."""
    print(f'{"phase":6s} {"runs":>6s} {"done":>6s} {"needs eval":>11s} {"unfinished":>11s} {"not started":>12s}  trained')
    total_runs = total_done = 0
    for phase in ('P0', 'P1', 'P2'):
        if phase not in phases:
            continue
        c = phase_status(programme.phase_runs(phase, cells=cells), runs_root)
        total_runs += c['runs']
        total_done += c['done']
        print(f'{phase:6s} {c["runs"]:6,d} {c["done"]:6,d} {c["needs_eval"]:11,d} {c["unfinished"]:11,d} {c["not_started"]:12,d}  {c["trained_share"] * 100:5.1f}%')
    if any(p in phases for p in ('P3', 'P4')):
        print('P3, P4: the best cell and controllers are chosen from the P2 validation results once P2 is done')
    print(f'{total_done:,} of {total_runs:,} runs in P0-P2 are finished. Unfinished runs start again from step 0 (a run cannot resume from its checkpoints).')
    print(f'continue with:  {command}')
    return 0


def _command_line(phases, actions, states, workers, device, threads, top_cells) -> str:
    parts = ['python -m natural_rl run-all']
    if tuple(phases) != ('P0', 'P1', 'P2', 'P3', 'P4'):
        parts.append(f'--phases {",".join(phases)}')
    for flag, value in (('actions', actions), ('states', states), ('threads', threads)):
        if value:
            parts.append(f'--{flag} {value}')
    if workers not in (None, 'auto'):
        parts.append(f'--workers {workers}')
    if device not in (None, 'auto'):
        parts.append(f'--device {device}')
    if top_cells != 1:
        parts.append(f'--top-cells {top_cells}')
    return ' '.join(parts)


def run_all(phases=('P0', 'P1', 'P2', 'P3', 'P4'), actions=None, states=None, workers='auto', device='auto', threads=None,
            dry_run=False, top_cells: int = 1, with_eval: bool = True, stream=None, status=False) -> int:
    from .env.contract import backend_of
    from .eval import load_scalers
    from .registry import repo_root, results_root, reward_config
    runs_root = results_root() / 'runs'
    cells = programme.parse_cells(actions, states)
    again = _command_line(phases, actions, states, workers, device, threads, top_cells)
    if status:
        return show_status(phases, cells, runs_root, again)
    res = plan_resources(workers, device, threads)
    est = programme.programme_totals(cells=cells)
    selected = [p for p in ('P0', 'P1', 'P2', 'P3', 'P4') if p in phases]
    total_runs = sum(est[p]['runs'] for p in selected)
    total_tr = sum(est[p]['transitions'] for p in selected)
    print(f'natural_rl run-all: {len(cells)} cell(s) (action/state): ' + ', '.join(f'{a}/{s}' for a, s in cells))
    print(f'  phases {",".join(selected)}: {total_runs:,} runs, {total_tr / 1e6:,.0f}M transitions '
          f'(P3/P4 are counted for one cell until the best cell is chosen from P2)')
    print(f'  resources: {res.describe()}')
    videos = sum(est[p]['videos'] for p in selected)
    print(f'  storage: {videos:,} videos of about 17 MB = about {videos * 17e6 / 1e9:,.0f} GB of raw video in {results_root()}')
    if dry_run:
        for p in selected:
            print(f'  {p}: {est[p]["runs"]:,} runs, {est[p]["transitions"] / 1e6:,.0f}M transitions, {est[p]["videos"]:,} videos')
        return 0
    overall = Overall(runs_root, total_runs, total_tr, res, stream=stream)
    overall.again = again
    logs = results_root() / 'logs'
    failures = []

    def stage(name, runs):
        overall.stage = name
        overall.add(runs)
        jobs = stage_jobs(runs, runs_root, res, with_eval)
        overall.message(f'[stage {name}] {len(runs)} runs, {len(jobs)} with work left')
        failed = run_jobs(jobs, res, overall, logs) if jobs else []
        overall.draw(force=True)
        if failed:
            failures.extend(failed)
            overall.message(f'[stage {name}] {len(failed)} run(s) failed; fix the cause and run the same command again to continue')
        return not failed

    def lock_rewards(p0_ids):
        if reward_config().get('status') == 'locked':
            return True
        overall.message('[lock] reward coefficients from the P0 policies')
        return cli_call('calibrate', 'rewards', '--pilot-runs', ','.join(p0_ids)) == 0

    def lock_scalers(run_ids, backends):
        doc = load_scalers
        if all(doc(b) is not None for b in backends):
            return True
        overall.message(f'[lock] HKS scalers (one pooled fit) from {len(run_ids)} validation evaluations')
        return cli_call('calibrate', 'scalers', '--runs', ','.join(run_ids)) == 0

    backends = sorted({backend_of(a) for a, _ in cells})
    p0 = programme.phase_runs('P0')
    p1 = programme.phase_runs('P1', cells=cells)
    p2 = programme.phase_runs('P2', cells=cells)
    if 'P0' in phases:
        if not stage('P0', p0) or not lock_rewards([r['run_id'] for r in p0]):
            return 1
    scalers_ok = False
    if 'P1' in phases:
        if not lock_rewards([r['run_id'] for r in p0]):
            overall.message('[stop] the reward coefficients are not locked: run P0 first')
            return 1
        if not stage('P1', p1):
            return 1
        scalers_ok = lock_scalers([r['run_id'] for r in p1], backends)
        if not scalers_ok:
            overall.message('[note] the P1 validation episodes do not fix the HKS scalers (too few successes); the registered '
                            'escalation fits them from the P2 validation data after P2 (nothing is lost: raw descriptors are stored)')
    if 'P2' in phases:
        if not stage('P2', p2):
            return 1
        if not scalers_ok:
            if not lock_scalers([r['run_id'] for r in p2], backends):
                overall.message('[stop] the HKS scalers could not be locked from P2 either; see the message above')
                return 1
            for run in p2:                                    # scores come from the stored raw descriptors
                cli_call('analyse', '--run-id', run['run_id'])
    if 'P3' in phases or 'P4' in phases:
        from .selection import best_cells, p3_selection
        later, table = best_cells(cells, top=top_cells)
        overall.message('[select] cells by mean validation R-HKS of the P2 r7_balanced runs:\n' + table.to_string(index=False)
                        + '\n         chosen: ' + ', '.join(f'{a}/{s}' for a, s in later))
        if 'P3' in phases:
            policies, ptable = p3_selection(cell=later[0])
            overall.message('[select] P3 controllers: ' + ','.join(policies))
            if not stage('P3', programme.phase_runs('P3', policies, later)):
                return 1
        if 'P4' in phases and not stage('P4', programme.phase_runs('P4', cells=later)):
            return 1
    overall.draw(force=True)
    overall.message(f'[done] {overall.line()}')
    return 0 if not failures else 1
