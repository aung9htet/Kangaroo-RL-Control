"""Explanatory figures for TECHNICAL.md, drawn from the same definitions the code uses."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..encoding import positional_encoding
from ..mdp import VelocityBins
from .plots import GRID, INK, INK_2, MUTED, SERIES, SURFACE, plt, save


def _box(ax, xy, w, h, text, colour=SERIES[0], fill=0.12, size=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.02,rounding_size=0.03', linewidth=1.4,
                                edgecolor=colour, facecolor=(*plt.matplotlib.colors.to_rgb(colour), fill)))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center', fontsize=size, color=INK)


def _arrow(ax, a, b, text=None, offset=(0, 0.03), rad=0.0, colour=INK_2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12, linewidth=1.3, color=colour,
                                 connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1], text, ha='center', va='bottom',
                fontsize=8.5, color=INK_2)


def mdp_loop(path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    _box(ax, (0.36, 0.70), 0.28, 0.2, 'Agent (DecQN)\n$a_j=\\arg\\max_k U_j(s,k)$', SERIES[0])
    _box(ax, (0.04, 0.12), 0.34, 0.26,
         'Model  $T(q,a)$  (training)\n$\\hat q=\\mathrm{clip}(q+\\Delta t\\,v(a))$\ncollision $\\Rightarrow q\'=q,\\ c=1$\n'
         'MuJoCo kinematics, 16 copies', SERIES[2])
    _box(ax, (0.62, 0.12), 0.34, 0.26,
         'Simulator  (evaluation)\nKangaroo_Libero, 100 Hz PD\n5 ticks per decision\nobject appears at the goal', SERIES[1])
    _box(ax, (0.415, 0.36), 0.17, 0.13, 'reward\n$r(q\',c,a;g)$', MUTED, 0.08)
    _arrow(ax, (0.40, 0.70), (0.24, 0.39), 'action $a=(k_1..k_7)$', (-0.06, 0.02))
    _arrow(ax, (0.60, 0.70), (0.76, 0.39), 'same action', (0.09, 0.0))
    _arrow(ax, (0.24, 0.38), (0.42, 0.42), None)
    _arrow(ax, (0.76, 0.38), (0.585, 0.42), None)
    _arrow(ax, (0.50, 0.49), (0.50, 0.70), None)
    ax.text(0.515, 0.585, "state $s'=(\\phi(q'),\\phi(g))$\nand reward $r$", ha='left', va='center', fontsize=8.5,
            color=INK_2)
    ax.text(0.5, 0.03, 'The goal $g$ (where the object appears) never changes during an episode; '
            'the transition never depends on it.', ha='center', fontsize=8.5, color=INK_2)
    save(fig, path)


def network(path, joint_freqs=4, goal_freqs=4, hidden=256):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0, 1)
    ax.axis('off')
    jw, gw = 7 * (2 * joint_freqs + 1), 3 * (2 * goal_freqs + 1)
    _box(ax, (0.01, 0.58), 0.14, 0.28, f'joint angles $q$ (7)\n$\\to$ $\\phi(q)$ ({jw})', SERIES[0], 0.08, 8.5)
    _box(ax, (0.01, 0.14), 0.14, 0.28, f'goal $g$ (3)\n$\\to$ $\\phi(g)$ ({gw})', SERIES[1], 0.08, 8.5)
    _box(ax, (0.21, 0.30), 0.13, 0.4, f'Linear {jw + gw}$\\to${hidden}\nLayerNorm\nReLU', MUTED, 0.06, 8.5)
    _box(ax, (0.38, 0.30), 0.13, 0.4, f'Linear {hidden}$\\to${hidden}\nLayerNorm\nReLU', MUTED, 0.06, 8.5)
    _box(ax, (0.55, 0.30), 0.13, 0.4, f'Linear {hidden}$\\to$49\nreshape 7$\\times$7\n$U_j(s,k)$', SERIES[2], 0.1, 8.5)
    _box(ax, (0.73, 0.55), 0.25, 0.3, '$Q(s,a)=\\frac{1}{7}\\sum_j U_j(s,a_j)$\n(value of a joint action)', SERIES[0], 0.08, 8.5)
    _box(ax, (0.73, 0.12), 0.25, 0.3, '$a^*_j=\\arg\\max_k U_j(s,k)$\nexact argmax over $7^7$ actions', SERIES[0], 0.08, 8.5)
    for y in (0.72, 0.28):
        _arrow(ax, (0.15, y), (0.21, 0.5))
    _arrow(ax, (0.34, 0.5), (0.38, 0.5))
    _arrow(ax, (0.51, 0.5), (0.55, 0.5))
    _arrow(ax, (0.68, 0.55), (0.73, 0.70))
    _arrow(ax, (0.68, 0.45), (0.73, 0.27))
    ax.text(0.08, 0.92, 'positional encoding computed inside the network', ha='center', fontsize=8, color=INK_2)
    save(fig, path)


def positional_encoding_figure(path, lower, upper, home, names, n_freq=4):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={'width_ratios': [1.25, 1]})
    x = np.linspace(-1, 1, 600)
    features = positional_encoding(x[:, None], n_freq)
    ax = axes[0]
    ax.plot(x, features[:, 0], color=INK, label='$\\tilde q$')
    for k in range(n_freq):
        colour = SERIES[k]
        ax.plot(x, features[:, 1 + 2 * k], color=colour, label=f'sin($2^{k}\\pi\\tilde q$)')
        ax.plot(x, features[:, 2 + 2 * k], color=colour, linestyle='--', linewidth=1.2, label=f'cos($2^{k}\\pi\\tilde q$)')
    ax.set_xlabel('normalised joint angle $\\tilde q = 2(q-q_{min})/(q_{max}-q_{min})-1$')
    ax.set_title('The 9 features of one joint')
    ax.legend(ncol=3, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.17))
    q = 2 * (np.asarray(home) - lower) / (upper - lower) - 1
    code = positional_encoding(q[None], n_freq)[0].reshape(2 * n_freq + 1, 7).T
    ax = axes[1]
    image = ax.imshow(code, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_yticks(range(7))
    ax.set_yticklabels([n.replace('_joint', '').replace('hand_right_', '') for n in names], fontsize=8)
    ax.set_xticks(range(2 * n_freq + 1))
    ax.set_xticklabels(['$\\tilde q$'] + [f'{f}{k}' for k in range(n_freq) for f in ('s', 'c')], fontsize=8)
    ax.set_title('State at the home pose: 7 x 9 = 63 numbers')
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.04)
    save(fig, path)


def action_lattice(path, bins: VelocityBins, spacing=0.005):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1, 1.3]})
    ax = axes[0]
    levels = np.asarray(bins.levels) * bins.v_max
    ax.bar(range(bins.n_bins), levels, color=SERIES[0], width=0.6)
    for i, v in enumerate(levels):
        ax.text(i, v + (0.05 if v >= 0 else -0.05), f'{v:+.1f} rad/s\n{v * bins.dt * 1000:+.0f} mrad',
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=7.5, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(bins.n_bins))
    ax.set_xlabel('bin index $k$ (same for every joint)')
    ax.set_ylabel('joint velocity (rad/s)')
    ax.set_ylim(-1.45, 1.45)
    ax.set_title('Velocity bins (step per decision, $\\Delta t$ = 0.05 s)')
    ax = axes[1]
    steps = np.round(levels * bins.dt / spacing).astype(int)
    start = np.array([12, 14])
    wall = lambda i, j: (i >= 20) & (j <= 9)               # an illustrative collision region
    n = 28
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    ax.scatter(ii[~wall(ii, jj)], jj[~wall(ii, jj)], s=3, color=GRID)
    ax.scatter(ii[wall(ii, jj)], jj[wall(ii, jj)], s=6, color=SERIES[7], alpha=0.5)
    for a in steps:
        for b in steps:
            target = np.clip(start + [a, b], 0, n - 1)
            rejected = wall(*target)
            end = start if rejected else target
            colour = SERIES[7] if rejected else (SERIES[1] if (target != start + [a, b]).any() else SERIES[0])
            if (end == start).all() and not rejected:
                continue
            ax.annotate('', xy=end if not rejected else target, xytext=start,
                        arrowprops=dict(arrowstyle='-|>', color=colour, lw=0.8, alpha=0.8, linestyle='-' if not rejected else ':'))
    ax.scatter(*start, s=40, color=INK, zorder=3)
    ax.axvline(n - 1, color=SERIES[1], linewidth=1.2)
    ax.text(n - 1.4, n - 1, 'joint limit:\nclipped', ha='right', va='top', fontsize=8, color=INK_2)
    ax.text(19.3, 3, 'collision region:\nmove rejected,\n$q\'=q$, $c=1$', ha='right', fontsize=8, color=INK_2)
    ax.set_xlabel('joint 1 (lattice index, 0.005 rad)')
    ax.set_ylabel('joint 2 (lattice index)')
    ax.set_title('49 transitions from one state (2 joints shown)')
    ax.set_aspect('equal')
    save(fig, path)


def her_figure(path, path_points, goal, picked=(8, 30, 80)):
    """Reach-point path of one episode (x-z), its goal, and the hindsight goals taken from it."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(path_points[:, 0], path_points[:, 2], color=SERIES[0], label='reach-point path of an episode')
    ax.scatter(*path_points[0, [0, 2]], color=INK, s=25, zorder=3, label='start')
    ax.scatter(goal[0], goal[2], marker='*', s=160, color=SERIES[7], zorder=3, label='goal $g$ (not reached)')
    for t in picked:
        p = path_points[t]
        ax.scatter(p[0], p[2], s=55, facecolor=SURFACE, edgecolor=SERIES[1], linewidth=2, zorder=3)
        ax.annotate(f'$\\tilde g=p(q_{{{t}}})$', p[[0, 2]], xytext=(6, 6), textcoords='offset points', fontsize=8,
                    color=INK_2)
    ax.scatter([], [], s=55, facecolor=SURFACE, edgecolor=SERIES[1], linewidth=2, label='hindsight goals $\\tilde g$')
    ax.set_xlabel('x (m, forward)')
    ax.set_ylabel('z (m, up)')
    ax.set_title('Hindsight relabelling: the same transitions, other goals')
    ax.legend(fontsize=8, loc='best')
    ax.set_aspect('equal')
    save(fig, path)
