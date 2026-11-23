#!/usr/bin/env python3
"""Per-layer download progress of an `enroot import` that is running inside a Slurm job.

    python3 slurm/image_progress.py 4746          # once;  add --watch to refresh every 5 s (Ctrl+C to leave)

Enroot writes each layer to /shared/enroot/cache/user-<uid>/<layer-digest>.<suffix> on the node the job runs on and the file grows as it
downloads, so this lists those files through `srun --overlap` and compares them with the layer sizes of the image (slurm/image_layers.json)."""
import json, subprocess, sys, time
from pathlib import Path

LAYERS = json.loads((Path(__file__).parent / 'image_layers.json').read_text())
CMD = 'for f in /shared/enroot/cache/*/*; do [ -f "$f" ] && stat -c "%s %n" "$f"; done 2>/dev/null; echo NODE $(hostname)'


def show(job):
    out = subprocess.run(['srun', f'--jobid={job}', '--overlap', '-N1', '-n1', 'sh', '-c', CMD], capture_output=True, text=True, timeout=60).stdout
    got, node = {}, '?'
    for line in out.splitlines():
        if line.startswith('NODE'):
            node = line.split()[1]
            continue
        size, path = line.split(' ', 1)
        digest = Path(path).name.split('.')[0]
        if digest in LAYERS:
            got[digest] = max(got.get(digest, 0), int(size))
    print(f'image download, job {job} on {node}   {time.strftime("%H:%M:%S")}\n')
    total = sum(LAYERS.values())
    have = 0
    for digest, size in sorted(LAYERS.items(), key=lambda kv: -kv[1]):
        if size < 1e6:
            continue
        g = min(got.get(digest, 0), size)
        have += g
        bar = '#' * int(20 * g / size) + '-' * (20 - int(20 * g / size))
        print(f'  {digest[:10]}  [{bar}] {100 * g / size:5.1f}%  {g / 1e6:7.1f} / {size / 1e6:7.1f} MB')
    print(f'\n  total {have / 1e6:,.0f} of {total / 1e6:,.0f} MB ({100 * have / total:.1f}%)   (a layer that is not listed yet, or finished and unpacked, shows 0 or 100)')


if __name__ == '__main__':
    job = next(a for a in sys.argv[1:] if not a.startswith('-'))
    if '--watch' in sys.argv:
        try:
            while True:
                print('\033[2J\033[H', end='')
                show(job)
                time.sleep(5)
        except KeyboardInterrupt:
            pass
    else:
        show(job)
