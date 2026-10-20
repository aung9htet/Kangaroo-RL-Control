"""Measure the facts the model relies on and write docs/results/probe.json plus figures.

    ./run_rl_docker.bash --headless -- python scripts/probe_model.py

- joints, limits, home pose, actuator kinds and gains of the 7 controlled joints
- collision test: agreement with full contact detection, and its cost
- reach envelope of the reach point against the table, for the stock and the chosen robot placement
- step and ramp responses of each joint in the simulator, with and without velocity feed-forward
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kangaroo_rl.config import load_config  # noqa: E402
from kangaroo_rl.envs.libero import LiberoReachEnv  # noqa: E402
from kangaroo_rl.experiment import make_bins, make_experiment, make_reward, repo_root  # noqa: E402
from kangaroo_rl.scene import ArmKinematics, build_scene  # noqa: E402
from kangaroo_rl.viz.plots import GRID, INK_2, SERIES, plt, save  # noqa: E402

FIGURES = repo_root() / 'docs' / 'figures'
RESULTS = repo_root() / 'docs' / 'results'


def envelope(scene, samples=200_000, seed=0):
    kinematics = ArmKinematics(scene)
    rng = np.random.default_rng(seed)
    points = np.array([kinematics.reach_point(rng.uniform(scene.lower, scene.upper)) for _ in range(samples)])
    return points


def table_band(scene, points, height, slab=0.02, margin=0.08):
    """Depth of the table strip (beyond its edge) where the reach point attains `height` above the top."""
    x0, x1, y0, y1, top = scene.table_box_base()
    sel = (np.abs(points[:, 2] - (top + height)) < slab) & (points[:, 0] > x0 + margin) & (points[:, 1] > y0) & (points[:, 1] < y1)
    return (float(points[sel, 0].max() - x0) if sel.any() else 0.0), int(sel.sum())


def responses(exp, feedforward):
    """Per joint: hold home, then a +0.3 rad/s ramp for 1 s, then hold. Returns time, reference, measured."""
    env = LiberoReachEnv(exp.scene, exp.bins, exp.reward, 100, mode='A', feedforward=feedforward,
                         target_object=None)
    up, zero = list(exp.bins.levels).index(0.3), exp.bins.zero
    out = []
    for j in range(exp.scene.n_joints):
        env.reset(exp.eval_goals(1)[0])
        ref, meas = [], []
        for t in range(40):
            action = np.full(7, zero)
            if 5 <= t < 25:
                action[j] = up
            env.step(action)
            ref.append(env.q_ref[j] - exp.scene.home[j])
            meas.append(env.measured()[j] - exp.scene.home[j])
        out.append((np.array(ref), np.array(meas)))
    return out


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_config(repo_root() / 'configs' / 'reach.yaml')
    exp = make_experiment(config, keep_env=True)
    scene = exp.scene
    physics = scene.env.sim.model._model
    report = {'joints': [], 'placement': {}, 'collision': {}, 'tracking': {}}
    for j, name in enumerate(scene.joint_names):
        channel = scene.channels[j]
        actuator = scene.env.robot.command_actuators[channel]
        gains = {'kp': float(scene.env.robot.kp[channel]), 'kd': float(scene.env.robot.kd[channel])} \
            if scene.kinds[j] == 'motor' else {'kp': float(physics.actuator_gainprm[actuator, 0]),
                                              'kv': float(-physics.actuator_biasprm[actuator, 2])}
        report['joints'].append({'name': name, 'kind': scene.kinds[j], 'lower': float(scene.lower[j]),
                                 'upper': float(scene.upper[j]), 'home': float(scene.home[j]), **gains,
                                 'torque_limit': float(physics.actuator_ctrlrange[actuator, 1])
                                 if scene.kinds[j] == 'motor' else None})

    # collision test cost
    kinematics = ArmKinematics(scene)
    rng = np.random.default_rng(0)
    qs = rng.uniform(scene.lower, scene.upper, (3000, 7))
    start = time.perf_counter()
    hits = sum(kinematics.evaluate(q)[1] for q in qs)
    report['collision'] = {'us_per_check': (time.perf_counter() - start) / len(qs) * 1e6,
                           'colliding_fraction_uniform': hits / len(qs), 'arm_geoms': int(len(scene.arm_geoms)),
                           'pairs_ignored_at_home': int(scene.ignored.sum() // 2)}
    start = time.perf_counter()
    for q in qs[:500]:
        kinematics.reach_point(q)
    report['collision']['us_per_fk'] = (time.perf_counter() - start) / 500 * 1e6

    # reach envelope, chosen placement and stock placement
    points = envelope(scene)
    stock = build_scene(scene.suite, scene.task_id, scene.seed, {'base_clearance': 0.55})
    stock_points = envelope(stock)
    aim = LiberoReachEnv(scene, exp.bins, exp.reward, 100, target_object=config['scene']['target_object'],
                         aim_offset=config['scene']['aim_offset'])
    aim_height = aim.object_rest_height() + aim.aim_offset[2] - scene.table_top_base
    for label, sc, pts in (('chosen', scene, points), ('stock', stock, stock_points)):
        depth, count = table_band(sc, pts, aim_height)
        report['placement'][label] = {'base_clearance': sc.settings['base_clearance'],
                                      'base_world': sc.base_pos.tolist(), 'table_edge_base_x': sc.table_edge_base,
                                      'object_band_depth_m': depth, 'samples_in_band': count}
    report['goal_pool'] = {'points': len(exp.pool), 'box_lower': exp.pool.lower.tolist(),
                           'box_upper': exp.pool.upper.tolist(), 'aim_height_above_top': aim_height}

    # workspace figure: side and top views
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (i, j, xl, yl) in zip(axes, [(0, 2, 'x (m, forward)', 'z (m, up)'), (0, 1, 'x (m, forward)', 'y (m, left)')]):
        for label, sc, pts, colour in (('stock placement (0.55 m)', stock, stock_points, SERIES[1]),
                                       ('chosen placement (0.25 m)', scene, points, SERIES[0])):
            shift = sc.base_pos - scene.base_pos        # draw both in the chosen base frame
            sub = pts[np.random.default_rng(1).choice(len(pts), 6000, replace=False)] + shift
            ax.scatter(sub[:, i], sub[:, j], s=1.2, color=colour, alpha=0.3, linewidths=0, label=label, rasterized=True)
        x0, x1, y0, y1, top = scene.table_box_base()
        if j == 2:
            ax.fill_between([x0, x1], top - 0.05, top, color=INK_2, alpha=0.5, linewidth=0, label='table top')
        else:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK_2, linewidth=1.5))
        lo, hi = exp.pool.lower, exp.pool.upper
        ax.add_patch(plt.Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], fill=False, edgecolor=SERIES[7],
                                   linewidth=1.5, linestyle='--', label='goal region'))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect('equal')
        ax.set_xlim(-0.9, 1.3)
    axes[0].set_title('Reach-point workspace, side view (robot base frame)')
    axes[1].set_title('Top view')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=8.5, markerscale=6,
               bbox_to_anchor=(0.5, -0.06))
    save(fig, FIGURES / 'workspace.png')

    # step / ramp responses
    dt = exp.bins.dt
    with_ff, without_ff = responses(exp, True), responses(exp, False)
    t = dt * np.arange(1, 41)
    fig, axes = plt.subplots(1, 7, figsize=(15, 2.8), sharey=True)
    lag = {}
    for j, ax in enumerate(axes):
        ref, meas = with_ff[j]
        _, meas_noff = without_ff[j]
        ax.plot(t, ref, color=INK_2, linewidth=1.2, linestyle='--', label='command (q_ref)')
        ax.plot(t, meas_noff, color=SERIES[1], label='measured, no feed-forward')
        ax.plot(t, meas, color=SERIES[0], label='measured, feed-forward')
        ax.set_title(scene.joint_names[j].replace('_joint', '').replace('hand_right_', ''), fontsize=9)
        ax.set_xlabel('time (s)')
        lag[scene.joint_names[j]] = {'ff_max_error': float(np.max(np.abs(meas - ref))),
                                     'noff_max_error': float(np.max(np.abs(meas_noff - ref))),
                                     'ff_final_error': float(abs(meas[-1] - ref[-1])),
                                     'noff_final_error': float(abs(meas_noff[-1] - ref[-1]))}
    axes[0].set_ylabel('joint angle - home (rad)')
    axes[-1].legend(loc='lower right', fontsize=7.5)
    fig.suptitle('Simulator tracking of a 0.3 rad/s ramp (1 s), per joint', fontsize=11, fontweight='bold', y=1.04)
    save(fig, FIGURES / 'step_response.png')
    report['tracking'] = lag
    (RESULTS / 'probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
