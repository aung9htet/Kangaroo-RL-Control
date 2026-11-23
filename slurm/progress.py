#!/usr/bin/env python3
"""Progress bar for the natural_rl programme. Plain Python 3, no container needed: run it on the manager after SSH.

    python3 slurm/progress.py            # show once
    python3 slurm/progress.py --watch    # refresh every 10 s (Ctrl+C to leave; the jobs keep running)

It reads results/runs/*/*/*/<run-id>/manifest.json (finished runs) and progress.json (live step of running ones), the same
files `run-all` uses, and compares them with the programme's size (P0-P2 in full; P3 and P4 for one cell: 2,901M transitions)."""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOTAL_RUNS, TOTAL_TR = 4053, 2_901_000_000
STAGES = [('P0', 3, 3_000_000), ('P1', 1440, 288_000_000), ('P2', 2400, 2_400_000_000), ('P3', 200, 200_000_000), ('P4', 10, 10_000_000)]


def scan():
    done = running = 0
    tr = 0.0
    first = None
    for m in (ROOT / 'results' / 'runs').glob('*/*/*/*/manifest.json'):
        try:
            doc = json.loads(m.read_text())
            steps = int(doc.get('steps', 0))
            first = m.stat().st_mtime if first is None else min(first, m.stat().st_mtime)
            if doc.get('status') == 'complete':
                done += 1
                tr += steps
                continue
            p = json.loads((m.parent / 'progress.json').read_text())
            running += 1
            tr += min(steps, steps * p['step'] / max(1, p['total']))
        except (OSError, ValueError, KeyError):
            pass
    return done, running, tr, first


def bar(frac, width=40):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return '[' + '#' * n + '-' * (width - n) + f'] {frac * 100:5.1f}%'


def hms(s):
    s = int(s)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    return (f'{d}d ' if d else '') + f'{h}h {s // 60:02d}m'


def show():
    done, running, tr, first = scan()
    print(f'natural_rl programme  {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    print('transitions ' + bar(tr / TOTAL_TR) + f'   {tr:,.0f} of {TOTAL_TR:,}')
    print('runs        ' + bar(done / TOTAL_RUNS) + f'   {done:,} of {TOTAL_RUNS:,} finished, {running} running now')
    if first and tr > 0:
        elapsed = time.time() - first
        speed = tr / elapsed
        left = (TOTAL_TR - tr) / speed
        print(f'\nspeed       {speed / 1e3:,.0f}k transitions/s on average over {hms(elapsed)}; about {hms(left)} left (rough: cells differ in speed)')
    print('\nJobs:')
    try:
        out = subprocess.run(['squeue', '-h', '-u', __import__('getpass').getuser(), '-o', '%i|%j|%t|%M|%R'],
                             capture_output=True, text=True, timeout=15).stdout
        rows = [line.split('|') for line in out.splitlines() if line.count('|') == 4]
        running = [r for r in rows if r[2] == 'R']
        pending = [r for r in rows if r[2] != 'R']
        for jid, name, _, elapsed, where in running:
            print(f'  running  {jid:>6s}  {name:<28s} {elapsed:>10s}  on {where}')
        if not running:
            print('  nothing running')
        if pending:
            nxt = sorted(pending, key=lambda r: int(r[0]))[0]
            print(f'  waiting  {len(pending)} job(s) queued behind them (next: {nxt[1]}, {nxt[4]});  full list: squeue -u $USER')
    except (OSError, subprocess.SubprocessError):
        print('  (squeue not available here)')
    print('\nstages of the programme: ' + ', '.join(f'{n} {r:,} runs' for n, r, _ in STAGES))


if __name__ == '__main__':
    if '--watch' in sys.argv:
        import contextlib, io
        previous = 0
        try:
            while True:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    show()
                    print('\n(refreshing every 10 s, Ctrl+C to leave; jobs keep running)')
                lines = buf.getvalue().splitlines()
                up = f'\033[{previous}A\r' if previous else ''       # back up over the last frame and overwrite it in place
                sys.stdout.write(up + ''.join(line + '\033[K\n' for line in lines) + '\033[J')
                sys.stdout.flush()
                previous = len(lines)
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    else:
        show()
