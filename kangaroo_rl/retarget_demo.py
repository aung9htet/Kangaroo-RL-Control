"""Sequential retargeting in the simulator: the object reappears somewhere else every few seconds and the
trained policy re-aims without any reset.

    ./run_rl_docker.bash --headless -- python -m kangaroo_rl.retarget_demo --run runs/reach/seed0

Writes docs/results/figures/retarget_sequence.gif, retarget_filmstrip.png and retarget_error.png, and the
per-step errors to docs/results/data/retarget.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_config
from .evaluate import cached_object_goals, load_agent, physics_env
from .experiment import EVAL_SEED, make_experiment, repo_root
from .viz.plots import GRID, INK_2, SERIES, plt, save
from .viz.render import Renderer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='final.pt')
    parser.add_argument('--targets', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=3.0, help='time per target')
    parser.add_argument('--mode', default='B')
    args = parser.parse_args(argv)
    config = load_config(args.run / 'config.yaml')
    exp = make_experiment(config, keep_env=True)
    agent, state = load_agent(args.run / 'checkpoints' / args.checkpoint)
    env = physics_env(exp, args.mode)
    pool = cached_object_goals(exp, env, 64)
    goals = pool[np.random.default_rng(EVAL_SEED + 7).choice(len(pool), args.targets, replace=False)]
    steps = int(round(args.seconds / exp.bins.dt))
    renderer = Renderer(env.model, env.data, width=720, height=540, view='reach')
    frames, errors, path, keyframes = [], [], [], []
    q, g = env.reset(goals[0])
    for k, goal in enumerate(goals):
        if k:
            q, g = env.retarget(goal)
        for t in range(steps):
            step, _ = env.step(agent.act(q[None], g[None])[0])
            q = step.q_next
            errors.append((k, t, step.distance))
            hand = exp.scene.to_world(env.reach_point())
            path.append(hand)
            if t % 2 == 0:
                frame = renderer.render(goals_world=[exp.scene.to_world(goal)], path_world=np.array(path[-steps:]),
                                        hand_world=hand)
                frames.append(frame)
            if t == steps - 1:
                keyframes.append(renderer.render(goals_world=[exp.scene.to_world(goal)],
                                                 path_world=np.array(path[-steps:]), hand_world=hand))
    figures = repo_root() / 'docs' / 'results' / 'figures'
    data = repo_root() / 'docs' / 'results' / 'data'
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    # every 4th decision at 400 x 300 with a 96-colour palette keeps the GIF at a few MB
    images = [Image.fromarray(f).resize((400, 300)).quantize(colors=96, method=Image.MEDIANCUT) for f in frames[::2]]
    images[0].save(figures / 'retarget_sequence.gif', save_all=True, append_images=images[1:],
                   duration=int(4 * exp.bins.dt * 1000), loop=0, optimize=True)
    fig, axes = plt.subplots(1, len(keyframes), figsize=(3.2 * len(keyframes), 2.7))
    for k, (ax, frame) in enumerate(zip(np.atleast_1d(axes), keyframes)):
        ax.imshow(frame)
        ax.set_axis_off()
        final = [e for e in errors if e[0] == k][-1][2]
        ax.set_title(f'target {k + 1}: {final * 100:.1f} cm after {args.seconds:.0f} s', fontsize=9)
    save(fig, figures / 'retarget_filmstrip.png')
    distance = np.array([e[2] for e in errors])
    time = exp.bins.dt * np.arange(1, len(distance) + 1)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(time, distance * 100, color=SERIES[0])
    for k in range(1, len(goals)):
        ax.axvline(k * args.seconds, color=INK_2, linewidth=0.8, linestyle='--')
    ax.axhline(2, color=SERIES[1], linewidth=1, linestyle=':')
    ax.text(time[-1], 2.3, '2 cm', ha='right', fontsize=8, color=INK_2)
    ax.set_yscale('log')
    ax.set_xlabel('time (s); dashed lines: the object reappears elsewhere')
    ax.set_ylabel('reach-point error (cm)')
    ax.set_title(f'Sequential retargeting in the simulator (mode {args.mode}, {state["step"]:,} training steps)')
    save(fig, figures / 'retarget_error.png')
    np.savetxt(data / 'retarget.csv', np.array(errors), delimiter=',', header='target,step,distance_m', comments='',
               fmt=['%d', '%d', '%.6f'])
    renderer.close()
    print('final errors per target (cm):', [round([e for e in errors if e[0] == k][-1][2] * 100, 2)
                                            for k in range(len(goals))])


if __name__ == '__main__':
    main()
