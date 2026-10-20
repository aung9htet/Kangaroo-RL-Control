"""Build every figure and table of the docs from the run directories.

    ./run_rl_docker.bash --headless -- python scripts/make_figures.py [--only learning dp ...]

Technical figures go to docs/figures/, results to docs/results/figures/ and tables to docs/results/data/.
A figure whose inputs do not exist yet is skipped with a note.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kangaroo_rl.config import load_config  # noqa: E402
from kangaroo_rl.experiment import repo_root  # noqa: E402
from kangaroo_rl.viz import diagrams  # noqa: E402
from kangaroo_rl.viz.plots import ENTITY, INK, INK_2, MUTED, SEQUENTIAL, SERIES, band, plt, save  # noqa: E402

ROOT = repo_root()
RUNS = ROOT / 'runs'
FIG = ROOT / 'docs' / 'figures'
RES = ROOT / 'docs' / 'results' / 'figures'
DATA = ROOT / 'docs' / 'results' / 'data'
CHECKPOINTS = ['step_00010000', 'step_00030000', 'step_00100000', 'step_00300000', 'step_01000000']


def read_metrics(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for mode in ('home', 'chain'):
        sel = [r for r in rows if r['mode'] == mode]
        out[mode] = {k: np.array([float(r[k]) for r in sel]) for k in sel[0] if k != 'mode'} if sel else {}
    return out


def seeds(name):
    return sorted(p for p in (RUNS / name).glob('seed*') if (p / 'metrics.csv').exists())


def baseline(name, kind='kinematic', mode='chain'):
    files = sorted((RUNS / 'baselines' / 'reach' / name / 'eval').glob(f'{kind}-{mode}-*.json'))
    return json.loads(files[0].read_text()) if files else None


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


# ------------------------------------------------------------------------------------------- results
def learning():
    groups = [('reach', 'RL (sin/cos)'), ('reach_raw', 'RL (raw angles)'), ('reach_noher', 'RL (no HER)')]
    available = [(n, l, seeds(n)) for n, l in groups if seeds(n)]
    if not available:
        return print('skip learning: no runs')
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for name, label, runs in available:
        curves = [read_metrics(r / 'metrics.csv')['chain'] for r in runs]
        n = min(len(c['step']) for c in curves)
        steps = curves[0]['step'][:n]
        band(axes[0], steps, [c['final_median'][:n] for c in curves], ENTITY[label], label, scale=100)
        band(axes[1], steps, [c['success_2cm'][:n] for c in curves], ENTITY[label], label, scale=100,
             direct_label=False)
        for run, c in zip(runs, curves):
            for i in range(n):
                rows.append([name, run.name, int(c['step'][i]), int(c['updates'][i]), c['final_median'][i],
                             c['final_p90'][i], c['success_1cm'][i], c['success_2cm'][i], c['success_5cm'][i],
                             c['rejected'][i]])
    for key, label in (('dls', 'Jacobian, binned'), ('dls_continuous', 'Jacobian, continuous')):
        b = baseline(key)
        if b:
            value = max(b['final_median'] * 100, 0.07)
            axes[0].axhline(value, color=ENTITY[label], linewidth=1.2, linestyle='--')
            shown = f'{b["final_median"] * 100:.1f} cm' if b['final_median'] * 100 >= 0.07 else '< 0.1 cm'
            axes[0].text(2e4, value * 1.2, f'{label} (model-based): {shown}', fontsize=8, color=INK_2)
            axes[1].axhline(b['success_2cm'] * 100, color=ENTITY[label], linewidth=1.2, linestyle='--')
            axes[1].text(2e4, b['success_2cm'] * 100 + 1.5, f'{label} (model-based)', fontsize=8, color=INK_2)
    axes[0].set_yscale('log')
    axes[0].set_ylim(0.05, 150)
    axes[0].set_ylabel('final distance to the goal, median (cm)')
    axes[1].set_ylabel('goals reached within 2 cm (%)')
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlabel('environment steps')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 1e3:.0f}k'))
        top = ax.secondary_xaxis('top', functions=(lambda s: np.maximum(s - 10000, 0) / 4, lambda u: u * 4 + 10000))
        top.set_xlabel('gradient updates', color=INK_2, fontsize=8.5)
        top.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 1e3:.0f}k'))
    axes[0].set_title('Accuracy against training (model, 256 held-out goals, chained)')
    axes[1].set_title('Success rate against training')
    axes[1].legend(loc='center left')
    save(fig, RES / 'learning_curve.png')
    write_csv(DATA / 'learning.csv', ['config', 'seed', 'step', 'updates', 'final_median_m', 'final_p90_m',
                                      'success_1cm', 'success_2cm', 'success_5cm', 'rejected'], rows)


def checkpoint_table():
    """Accuracy at each saved checkpoint, model (home + chain) and simulator (A, B), every seed."""
    runs = seeds('reach')
    if not runs:
        return print('skip checkpoints: no runs')
    rows, dist = [], {}
    for run in runs:
        for name in CHECKPOINTS + ['final']:
            for kind in ('kinematic-home', 'kinematic-chain', 'liberoA-home', 'liberoB-home'):
                path = run / 'eval' / f'{kind}-{name}.json'
                if not path.exists():
                    continue
                s = json.loads(path.read_text())
                rows.append([run.name, name, s.get('step', ''), kind, s['final_median'], s['final_p90'], s['final_max'],
                             s['success_1cm'], s['success_2cm'], s['success_5cm'], s.get('hold_2cm', ''),
                             s.get('time_to_2cm_median', ''), s.get('tracking_median', ''), s.get('object_shift_median', ''),
                             s.get('object_shift_max', '')])
                d = np.load(path.with_suffix('.npz'))['distances'][:, -1]
                dist.setdefault((kind, name), []).append(d)
    write_csv(DATA / 'checkpoints.csv', ['seed', 'checkpoint', 'step', 'evaluation', 'final_median_m', 'final_p90_m',
                                         'final_max_m', 'success_1cm', 'success_2cm', 'success_5cm', 'hold_2cm',
                                         'time_to_2cm_s', 'tracking_median_rad', 'object_shift_median_m',
                                         'object_shift_max_m'], rows)
    kinds = [('kinematic-chain', 'model, chained (256 goals)', SERIES[0]),
             ('liberoA-home', 'simulator, mode A (object goals)', SERIES[1]),
             ('liberoB-home', 'simulator, mode B (object goals)', SERIES[2])]
    names = [n for n in CHECKPOINTS if any((k, n) in dist for k, _, _ in kinds)]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(11, 4.3))
    width = 0.26
    for i, (kind, label, colour) in enumerate(kinds):
        for j, name in enumerate(names):
            if (kind, name) not in dist:
                continue
            values = np.concatenate(dist[(kind, name)]) * 100
            pos = j + (i - 1) * width
            parts = ax.boxplot(values, positions=[pos], widths=width * 0.8, patch_artist=True, showfliers=False,
                               medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=colour),
                               capprops=dict(color=colour), boxprops=dict(facecolor=colour, edgecolor=colour, alpha=0.55))
        ax.plot([], [], color=colour, linewidth=8, alpha=0.55, label=label)
    ax.axhline(2, color=MUTED, linewidth=1, linestyle=':')
    ax.text(len(names) - 0.55, 2.2, '2 cm', fontsize=8, color=INK_2)
    ax.set_yscale('log')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f'{int(n.split("_")[1]) // 1000}k steps' for n in names])
    ax.set_ylabel('final distance to the goal (cm)')
    ax.set_title('Final error at each checkpoint, all seeds (box: quartiles, whiskers: 1.5 IQR)')
    ax.legend(loc='upper right')
    save(fig, RES / 'accuracy_vs_iterations.png')


def error_map():
    run = RUNS / 'reach' / 'seed0'
    path = run / 'eval' / 'kinematic-chain-final.npz'
    if not path.exists():
        path = next(iter(sorted((run / 'eval').glob('kinematic-chain-step_*.npz'))[-1:]), None)
    if path is None or not path.exists():
        return print('skip error map')
    data = np.load(path)
    final = data['distances'][:, -1] * 100
    goals = data['goals']
    config = load_config(run / 'config.yaml')
    probe = json.loads((ROOT / 'docs/results/probe.json').read_text())
    edge = probe['placement']['chosen']['table_edge_base_x']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (i, j, xl, yl) in zip(axes, [(0, 1, 'x (m, forward)', 'y (m, left)'), (0, 2, 'x (m, forward)', 'z (m, up)')]):
        sc = ax.scatter(goals[:, i], goals[:, j], c=np.clip(final, 0.1, 30), cmap=SEQUENTIAL, s=26,
                        norm=plt.matplotlib.colors.LogNorm(0.1, 30), edgecolors=MUTED, linewidths=0.3)
        ax.axvline(edge, color=INK_2, linewidth=1, linestyle='--')
        ax.text(edge + 0.005, ax.get_ylim()[1], 'table edge', fontsize=8, color=INK_2, va='top')
        ax.scatter([0], [0], marker='s', color=INK, s=30)
        ax.annotate('robot base', (0, 0), xytext=(4, 4), textcoords='offset points', fontsize=8, color=INK_2)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect('equal')
    fig.colorbar(sc, ax=axes, fraction=0.025, label='final error (cm, log scale)')
    axes[0].set_title('Where the error is: held-out goals, top view')
    axes[1].set_title('Side view')
    save(fig, RES / 'error_map.png')


def dp_figures():
    out = RUNS / 'slice2' / 'seed0' / 'dp'
    if not (out / 'dp_check.json').exists():
        return print('skip dp: no dp_check')
    report = json.loads((out / 'dp_check.json').read_text())
    data = np.load(out / 'goal0.npz')
    value, learned, collides = data['value'], data['learned'], data['collides']
    axis0, axis1 = data['axis0'], data['axis1']
    extent = [axis1[0], axis1[-1], axis0[0], axis0[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={'wspace': 0.32})
    masked = lambda a: np.ma.masked_where(collides, a)
    vmin, vmax = value[~collides].min(), value[~collides].max()
    for ax, array, title in ((axes[0], value, '$V^*$ from value iteration (exact)'),
                             (axes[1], learned, 'learned value, max over actions of $Q_\\theta$')):
        image = ax.imshow(masked(array), origin='lower', extent=extent, aspect='auto', cmap=SEQUENTIAL,
                          vmin=vmin, vmax=vmax)
        ax.set_xlabel('elbow arm_right_4 (rad)')
        ax.set_ylabel('shoulder pitch arm_right_1 (rad)')
        ax.set_title(title)
        ax.grid(False)
    fig.colorbar(image, ax=axes[1], fraction=0.05, pad=0.03, label='value')
    shifts = data['shifts']
    dp_policy, rl_policy = data['dp_policy'], data['rl_policy']
    step = 14
    i, j = np.meshgrid(np.arange(0, len(axis0), step), np.arange(0, len(axis1), step), indexing='ij')
    free = ~collides[i, j]
    ax = axes[2]
    ax.imshow(np.ma.masked_where(~collides, collides.astype(float)), origin='lower', extent=extent, aspect='auto',
              cmap='Reds', vmin=0, vmax=2.5)
    for policy, colour, label in ((dp_policy, SERIES[2], 'optimal (value iteration)'),
                                  (rl_policy, SERIES[0], 'learned (DecQN)')):
        di = shifts[policy[i, j, 0]][free] * 0.005
        dj = shifts[policy[i, j, 1]][free] * 0.005
        ax.quiver(axis1[j][free], axis0[i][free], dj, di, color=colour, angles='xy', scale_units='xy', scale=0.6,
                  width=0.004, alpha=0.85, label=label)
    goal_cell = np.unravel_index(np.argmin(np.linalg.norm(data['points'] - data['goal'], axis=-1)), collides.shape)
    ax.scatter(axis1[goal_cell[1]], axis0[goal_cell[0]], marker='*', s=180, color=SERIES[7], zorder=4, label='goal')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('elbow arm_right_4 (rad)')
    ax.set_title('Greedy actions (red: arm in collision)')
    ax.legend(loc='lower left', fontsize=7.5)
    ax.grid(False)
    save(fig, FIG / 'bellman_slice.png')
    # agreement over training + DP convergence
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for g, entry in enumerate(report['per_goal']):
        cks = sorted(entry['checkpoints'].values(), key=lambda m: m['step'])
        steps = [m['step'] for m in cks]
        axes[0].plot(steps, [100 * m['near_optimal_actions'] for m in cks], color=SERIES[g], marker='o', markersize=4,
                     label=f'goal {g + 1}')
        sweeps = sorted(int(s) for s in entry['dp_by_sweep'])
        axes[1].plot(sweeps, [100 * entry['dp_by_sweep'][str(s)]['final_p90'] for s in sweeps], color=SERIES[g],
                     marker='o', markersize=4, label=f'goal {g + 1}')
        for m in cks:
            rows.append([g + 1, m['step'], m['near_optimal_actions'], m['exact_actions'], m['value_correlation'],
                         m['value_mae'], m['dp_final_median'], m['rl_final_median'], m['rl_final_p90'], m['regret_mean']])
    axes[0].set_xscale('log')
    axes[0].set_xlabel('environment steps (DecQN, 2 joints)')
    axes[0].set_ylabel('states with a near-optimal greedy action (%)')
    axes[0].set_title('Learned policy against the exact optimum')
    axes[0].legend(fontsize=8)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('value-iteration sweeps')
    axes[1].set_ylabel('final distance, 90th percentile (cm)')
    axes[1].set_title('Exact solution: accuracy against Bellman sweeps')
    save(fig, RES / 'dp_check.png')
    write_csv(DATA / 'dp_check.csv', ['goal', 'step', 'near_optimal_actions', 'exact_actions', 'value_correlation',
                                      'value_mae', 'dp_final_median_m', 'rl_final_median_m', 'rl_final_p90_m',
                                      'regret_mean'], rows)


def baselines_table():
    rows = []
    for name in ('random', 'dls', 'dls_continuous'):
        for kind, mode in (('kinematic', 'home'), ('kinematic', 'chain'), ('liberoA', 'home'), ('liberoB', 'home')):
            b = baseline(name, kind, mode)
            if b:
                rows.append([name, f'{kind}-{mode}', b['final_median'], b['final_p90'], b['final_max'], b['success_2cm'],
                             b.get('rejected', '')])
    if rows:
        write_csv(DATA / 'baselines.csv', ['controller', 'evaluation', 'final_median_m', 'final_p90_m', 'final_max_m',
                                           'success_2cm', 'rejected'], rows)


# ------------------------------------------------------------------------------------------- technical
def technical():
    from kangaroo_rl.experiment import make_bins
    config = load_config(ROOT / 'configs' / 'reach.yaml')
    bins = make_bins(config)
    diagrams.mdp_loop(FIG / 'mdp_loop.png')
    diagrams.network(FIG / 'network.png')
    diagrams.action_lattice(FIG / 'action_lattice.png', bins)
    probe = json.loads((ROOT / 'docs/results/probe.json').read_text())
    lower = np.array([j['lower'] for j in probe['joints']])
    upper = np.array([j['upper'] for j in probe['joints']])
    home = np.array([j['home'] for j in probe['joints']])
    diagrams.positional_encoding_figure(FIG / 'positional_encoding.png', lower, upper, home,
                                        [j['name'] for j in probe['joints']])


def renders():
    """Scene pictures from the simulator: overview with the object and goal, the 7 joint axes, HER example,
    and the same goal reached by successive checkpoints."""
    from kangaroo_rl.evaluate import cached_object_goals, load_agent, physics_env
    from kangaroo_rl.experiment import make_experiment
    from kangaroo_rl.viz.render import Renderer, joint_overlay, project
    config = load_config(ROOT / 'configs' / 'reach.yaml')
    exp = make_experiment(config, keep_env=True)
    env = physics_env(exp, 'B')
    goals = cached_object_goals(exp, env, 64)
    goal = goals[0]
    env.reset(goal)
    renderer = Renderer(env.model, env.data, width=1100, height=760, view='overview')
    frame = renderer.render(goals_world=[exp.scene.to_world(goal)])
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.imshow(frame)
    ax.set_axis_off()
    for label, point in (('goal: aim point above the bowl', exp.scene.to_world(goal)),
                         ('reach point (palm)', exp.scene.to_world(env.reach_point()))):
        u, v = project(renderer, point)[0]
        ax.annotate(label, (u, v), xytext=(u + 70, v - 70), fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle='-', color=INK_2, lw=0.8),
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=MUTED, lw=0.6))
    ax.set_title('Kangaroo in the LIBERO scene (libero_spatial task 0), base 0.25 m behind the table', fontsize=10)
    save(fig, FIG / 'scene.png')
    renderer.set_view('side')
    image, anchors, colours = joint_overlay(renderer, exp.scene)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.imshow(image)
    ax.set_axis_off()
    pixels = project(renderer, anchors)
    for k, ((u, v), name) in enumerate(zip(pixels, exp.scene.joint_names)):
        ax.annotate(f'{k + 1}: {name.replace("_joint", "")}', (u, v), xytext=(u + 120, v - 160 + 48 * k), fontsize=8.5,
                    color=INK, arrowprops=dict(arrowstyle='-', color=colours[k][:3], lw=1.2),
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=colours[k][:3], lw=1))
    ax.set_title('The 7 controlled joints (axes drawn at their anchors); green: reach point', fontsize=10)
    save(fig, FIG / 'arm_joints.png')
    # same object goal across checkpoints (seed 0)
    run = RUNS / 'reach' / 'seed0' / 'checkpoints'
    names = [n for n in CHECKPOINTS if (run / f'{n}.pt').exists()]
    if names:
        renderer.set_view('reach')
        frames, finals = [], []
        target = goals[3]
        for name in names:
            agent, _ = load_agent(run / f'{name}.pt')
            q, g = env.reset(target)
            path = []
            for _ in range(exp.horizon):
                step, _ = env.step(agent.act(q[None], g[None])[0])
                q = step.q_next
                path.append(exp.scene.to_world(env.reach_point()))
            frames.append(renderer.render(goals_world=[exp.scene.to_world(target)], path_world=np.array(path),
                                          hand_world=path[-1]))
            finals.append(step.distance)
        fig, axes = plt.subplots(1, len(frames), figsize=(3.3 * len(frames), 2.9))
        for ax, frame, name, final in zip(np.atleast_1d(axes), frames, names, finals):
            ax.imshow(frame)
            ax.set_axis_off()
            ax.set_title(f'{int(name.split("_")[1]) // 1000}k steps: {final * 100:.1f} cm', fontsize=9)
        save(fig, RES / 'filmstrip_checkpoints.png')
        # HER illustration from an early checkpoint's path (it moves but rarely reaches its goal)
        agent, _ = load_agent(run / f'{names[min(2, len(names) - 1)]}.pt')
        q, g = env.reset(goals[5])
        path = []
        for _ in range(exp.horizon):
            step, _ = env.step(agent.act(q[None], g[None])[0])
            q = step.q_next
            path.append(env.reach_point())
        diagrams.her_figure(FIG / 'her.png', np.array(path), goals[5])
    renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    for d in (FIG, RES, DATA):
        d.mkdir(parents=True, exist_ok=True)
    steps = {'technical': technical, 'learning': learning, 'checkpoints': checkpoint_table, 'map': error_map,
             'dp': dp_figures, 'baselines': baselines_table, 'renders': renders}
    for name, function in steps.items():
        if args.only and name not in args.only:
            continue
        print(f'-- {name}', flush=True)
        function()


if __name__ == '__main__':
    main()
