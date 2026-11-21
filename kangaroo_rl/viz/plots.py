"""Static figures (PNG) for the docs, in one consistent style.

Colours follow a validated categorical order (blue, orange, aqua, ...) assigned to fixed entities:
the RL agent is always blue, the quantised Jacobian baseline orange, the continuous one aqua.
Magnitude maps use one hue, light to dark.
"""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK_2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
SEQUENTIAL = 'Blues'
ENTITY = {'RL (sin/cos)': SERIES[0], 'Jacobian, binned': SERIES[1], 'Jacobian, continuous': SERIES[2],
          'RL (raw angles)': SERIES[3], 'RL (no HER)': SERIES[4], 'Random': MUTED, 'Value iteration': SERIES[2]}


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'axes.labelsize': 9.5, 'xtick.color': INK_2, 'ytick.color': INK_2,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
        'legend.frameon': False, 'legend.fontsize': 8.5, 'font.size': 9.5, 'figure.dpi': 110,
        'savefig.dpi': 160, 'savefig.bbox': 'tight', 'text.color': INK,
    })


style()


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def band(ax, x, runs, colour, label, scale=1.0, direct_label=True):
    """Every seed as a thin line, the median across seeds as a 2 px line."""
    runs = np.asarray(runs) * scale
    for run in runs:
        ax.plot(x, run, color=colour, linewidth=0.8, alpha=0.35)
    median = np.median(runs, axis=0)
    ax.plot(x, median, color=colour, label=label)
    if direct_label:
        ax.annotate(label, (x[-1], median[-1]), xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize=8.5, color=INK_2)
    return median
