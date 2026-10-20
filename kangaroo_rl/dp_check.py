"""Check the learned Q-function against the exact Bellman solution on the 2-joint lattice.

    python -m kangaroo_rl.train --config configs/slice2.yaml --seed 0
    python -m kangaroo_rl.dp_check --run runs/slice2/seed0

For each of a few goals over the table: value iteration gives V* and Q*; every checkpoint of the run is
compared with it (near-optimal greedy actions, value correlation, returns from 500 starts). Writes
<run>/dp/dp_check.json and <run>/dp/goal<i>.npz (V*, the learned V and both policies, for the figures).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .dp import Lattice, build_lattice, compare, rollout, transition_tables, value_iteration
from .evaluate import load_agent
from .experiment import cache_dir, make_experiment

SNAPSHOTS = (1, 2, 5, 10, 20, 50, 100, 200, 500)


def cached_lattice(exp, spacing) -> Lattice:
    key = hashlib.sha1(f'{exp.scene.signature()}|{exp.active.tolist()}|{exp.limits.round(5)}|{spacing}|'
                       f'{exp.bins}'.encode()).hexdigest()[:12]
    path = cache_dir() / f'lattice-{key}.npz'
    if path.exists():
        data = np.load(path)
        return Lattice([data['axis0'], data['axis1']], data['points'], data['collides'], data['shifts'], exp.active)
    lattice = build_lattice(exp.scene, exp.active, exp.limits, spacing, exp.bins)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, axis0=lattice.axes[0], axis1=lattice.axes[1], points=lattice.points,
             collides=lattice.collides, shifts=lattice.shifts)
    return lattice


def choose_goals(exp, lattice, n, seed=0):
    """Spread-out goals among collision-free lattice reach points inside the goal box."""
    points = lattice.points[~lattice.collides]
    inside = np.all((points >= exp.pool.lower) & (points <= exp.pool.upper), axis=1)
    candidates = points[inside]
    rng = np.random.default_rng(seed)
    goals = [candidates[rng.integers(len(candidates))]]
    for _ in range(n - 1):
        distance = np.min(np.linalg.norm(candidates[:, None] - np.array(goals)[None], axis=-1), axis=1)
        goals.append(candidates[distance.argmax()])
    return np.array(goals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoints', nargs='*', default=['all'])
    args = parser.parse_args(argv)
    config = load_config(args.run / 'config.yaml')
    exp = make_experiment(config)
    spacing = float(config['dp']['spacing'])
    gamma = float(config['agent']['gamma'])
    lattice = cached_lattice(exp, spacing)
    goals = choose_goals(exp, lattice, int(config['dp']['n_goals']))
    names = sorted(p.name for p in (args.run / 'checkpoints').glob('step_*.pt')) \
        if args.checkpoints == ['all'] else args.checkpoints
    agents = {name: load_agent(args.run / 'checkpoints' / name) for name in names}
    out = args.run / 'dp'
    out.mkdir(exist_ok=True)
    report = {'lattice_shape': list(lattice.shape), 'spacing': spacing, 'free_states': int((~lattice.collides).sum()),
              'goals': goals.tolist(), 'per_goal': []}
    for g, goal in enumerate(goals):
        tables = transition_tables(lattice, exp.bins, exp.reward, goal)
        value, q_star, sweeps, snapshots = value_iteration(tables, gamma, snapshots=SNAPSHOTS)
        rng = np.random.default_rng(g)
        cells = np.argwhere(~lattice.collides)
        starts = cells[rng.choice(len(cells), 500, replace=False)]
        by_sweep = {}
        for sweep, policy in snapshots.items():
            final, _ = rollout(lattice, tables, policy, starts, goal, exp.horizon, gamma)
            by_sweep[sweep] = {'final_median': float(np.median(final)), 'final_p90': float(np.percentile(final, 90))}
        entry = {'goal': goal.tolist(), 'sweeps': sweeps, 'dp_by_sweep': by_sweep, 'checkpoints': {}}
        for name, (agent, state) in agents.items():
            metrics, actions, learned = compare(agent, lattice, tables, value, q_star, goal, exp.horizon, gamma,
                                                seed=g)
            metrics.update(step=state['step'], updates=state['updates'])
            entry['checkpoints'][name] = metrics
            print(f'goal {g} {name}: near-optimal actions {metrics["near_optimal_actions"]:.3f}  '
                  f'value r {metrics["value_correlation"]:.3f}  final median DP {metrics["dp_final_median"]*100:.2f} '
                  f'cm / RL {metrics["rl_final_median"]*100:.2f} cm', flush=True)
        last = names[-1] if names else None
        if last is not None:
            metrics, actions, learned = compare(agents[last][0], lattice, tables, value, q_star, goal, exp.horizon,
                                                gamma, seed=g)
            np.savez_compressed(out / f'goal{g}.npz', value=value, learned=learned, rl_policy=actions,
                                dp_policy=np.stack(np.unravel_index(
                                    q_star.reshape(-1, *lattice.shape).argmax(0), q_star.shape[:2]), axis=-1),
                                collides=lattice.collides, axis0=lattice.axes[0], axis1=lattice.axes[1],
                                points=lattice.points, goal=goal, shifts=lattice.shifts)
        report['per_goal'].append(entry)
    (out / 'dp_check.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
