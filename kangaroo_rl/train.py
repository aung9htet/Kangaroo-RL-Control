"""Train the goal-conditioned DecQN agent on the model MDP.

    python -m kangaroo_rl.train --config configs/reach.yaml --seed 0 [--set train.steps=100000]

Writes runs/<name>/seed<k>/: config.yaml, run.json, metrics.csv (evaluation every `eval_every`
environment steps, from home and chained), train.csv, checkpoints/step_<n>.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .agent import make_agent
from .agent.decqn import Explorer
from .agent.replay import EpisodeReplay
from .config import load_config, override
from .envs.kinematic import KinematicReachEnv
from .evaluation import KinematicEvaluator
from .experiment import make_experiment, repo_root
from .runinfo import run_metadata


def epsilon(step, cfg):
    fraction = min(1.0, step / max(1.0, cfg['eps_fraction'] * cfg['steps']))
    return cfg['eps_start'] + fraction * (cfg['eps_end'] - cfg['eps_start'])


def pick_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


class CsvLog:
    def __init__(self, path):
        self.path = Path(path)
        self.fields = None

    def write(self, row: dict):
        new = self.fields is None
        if new:
            self.fields = list(row)
        with self.path.open('a' if not new else 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            if new:
                writer.writeheader()
            writer.writerow(row)


def new_starts(env, starts, rng, mix, first=False):
    """Continuation (keep the pose: the next goal is a retarget), home with noise, or a random pose."""
    q = env.observe()[0]
    p_continue, p_home = (0.0, mix['home'] / (mix['home'] + mix['random'])) if first else \
        (mix['continuation'], mix['home'])
    for i in range(env.n):
        u = rng.random()
        if u < p_continue:
            continue
        q[i] = starts.home(rng) if u < p_continue + p_home else starts.random(rng)
    return q


def train(config: dict, seed: int, out: Path, device='auto', threads=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    tc = config['train']
    device = pick_device(device)
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    exp = make_experiment(config)
    scene, bins, reward = exp.scene, exp.bins, exp.reward
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    meta = run_metadata(scene)
    meta.update(seed=seed, device=device, goal_pool=len(exp.pool))
    (out / 'run.json').write_text(json.dumps(meta, indent=2, default=str))

    n_envs = int(tc['n_envs'])
    active, limits = exp.active, exp.limits
    n_joints = len(active)
    env = KinematicReachEnv(scene, n_envs, bins, reward, exp.horizon, config['mdp']['collision'],
                            float(config['mdp'].get('q_noise', 0.0)), active=active, limits=limits, seed=seed)
    starts = exp.starts()
    agent_cfg = config['agent']
    agent = make_agent(agent_cfg.get('kind', 'additive'), exp.network_config(), gamma=agent_cfg['gamma'], lr=float(agent_cfg['lr']),
                       tau=agent_cfg['tau'], grad_clip=agent_cfg['grad_clip'], reward_bounds=reward.bounds,
                       double=agent_cfg['double'], device=device)
    replay = EpisodeReplay(int(tc['replay']), exp.horizon, n_joints, reward, bins,
                           her_ratio=float(tc['her_ratio']), seed=seed, n_step=int(agent_cfg.get('n_step', 1)),
                           gamma=float(agent_cfg['gamma']))
    explorer = Explorer(n_envs, n_joints, bins.n_bins, int(tc['max_repeat']), seed=seed)
    evaluator = KinematicEvaluator(scene, bins, reward, exp.horizon, exp.eval_goals(), n_envs=32,
                                   check_collision=config['mdp']['collision'], active=active, limits=limits)
    metrics_log, train_log = CsvLog(out / 'metrics.csv'), CsvLog(out / 'train.csv')

    def evaluate(step, wall):
        for mode in ('home', 'chain'):
            summary, _ = evaluator.run(agent.act, mode)
            metrics_log.write({'step': step, 'updates': agent.updates, 'wall_s': round(wall, 1), 'mode': mode,
                               **{k: round(v, 6) for k, v in summary.items()}})
        return summary

    total, warmup = int(tc['steps']), int(tc['warmup'])
    checkpoints = sorted(int(c) for c in tc['checkpoints'])
    q, goal = env.reset_to(new_starts(env, starts, rng, config['starts'], first=True), exp.pool.sample(rng, n_envs))
    replay.begin(goal)
    start_time = time.time()
    next_eval, next_ckpt, losses = 0, 0, []
    step = 0
    while True:
        if step >= next_eval:
            summary = evaluate(step, time.time() - start_time)
            print(f'step {step:>8d}  updates {agent.updates:>7d}  chain median {summary["final_median"]*100:6.2f} cm  '
                  f'p90 {summary["final_p90"]*100:6.2f} cm  success@2cm {summary["success_2cm"]:.2f}  '
                  f'({time.time() - start_time:.0f} s)', flush=True)
            next_eval += int(tc['eval_every'])
        while next_ckpt < len(checkpoints) and step >= checkpoints[next_ckpt]:
            torch.save({**agent.state(), 'step': step, 'config': config}, out / 'checkpoints' / f'step_{checkpoints[next_ckpt]:08d}.pt')
            next_ckpt += 1
        if step >= total:
            break
        eps = epsilon(step, tc)
        greedy = agent.act(q, goal) if step >= warmup else np.zeros((n_envs, n_joints), dtype=np.int64)
        actions = explorer(greedy, eps if step >= warmup else 1.0)
        result, done = env.step(actions)
        replay.add(env.t - 1, q, actions, result.q_next, result.collided, result.point_next, result.point)
        q = result.q_next
        step += n_envs
        if done:
            replay.end()
            q, goal = env.reset_to(new_starts(env, starts, rng, config['starts']), exp.pool.sample(rng, n_envs))
            replay.begin(goal)
            explorer.reset()
        if step >= warmup and len(replay) > 0:
            for _ in range(int(tc['updates_per_step'])):
                losses.append(agent.update(replay.sample(int(agent_cfg['batch']))))
            if agent.updates % 1000 < int(tc['updates_per_step']):
                train_log.write({'step': step, 'updates': agent.updates, 'epsilon': round(eps, 4),
                                 'loss': np.mean([l['loss'] for l in losses]),
                                 'q_mean': np.mean([l['q_mean'] for l in losses]),
                                 'grad_norm': np.mean([l['grad_norm'] for l in losses]),
                                 'wall_s': round(time.time() - start_time, 1)})
                losses = []
    torch.save({**agent.state(), 'step': step, 'config': config}, out / 'checkpoints' / 'final.pt')
    meta.update(wall_s=round(time.time() - start_time, 1), updates=agent.updates, steps=step)
    (out / 'run.json').write_text(json.dumps(meta, indent=2, default=str))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/reach.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='override config values')
    args = parser.parse_args(argv)
    config = override(load_config(args.config), args.set)
    out = args.out or repo_root() / 'runs' / config['name'] / f'seed{args.seed}'
    train(config, args.seed, out, args.device, args.threads)


if __name__ == '__main__':
    main()
