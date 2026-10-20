"""Evaluate trained checkpoints or reference controllers, in the model or in the simulator.

    python -m kangaroo_rl.evaluate --run runs/reach/seed0 --backend kinematic
    python -m kangaroo_rl.evaluate --run runs/reach/seed0 --backend libero --modes A B --goals 64
    python -m kangaroo_rl.evaluate --config configs/reach.yaml --baseline dls --backend kinematic

Model backend: the 256 held-out goals of training, from home and chained. Simulator backend: an object
appears on the table and the goal is the aim point above it (Kangaroo_Libero physics, modes A and B).
Each result is written as JSON (summary) and NPZ (error after every step) under <run>/eval/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .agent import make_agent
from .baselines import DLSPolicy, RandomPolicy
from .config import load_config, override
from .envs.libero import LiberoReachEnv
from .evaluation import KinematicEvaluator, object_goals, physics_run
from .experiment import EVAL_SEED, cache_dir, make_experiment, repo_root


def load_agent(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    config = state['config']
    agent = make_agent(state.get('kind', 'additive'), state['network_config'], gamma=config['agent']['gamma'],
                       device=device)
    agent.load(state, weights_only=True)
    return agent, state


def physics_env(exp, mode):
    sc = exp.config['scene']
    return LiberoReachEnv(exp.scene, exp.bins, exp.reward, exp.horizon, mode=mode,
                          target_object=sc.get('target_object'), aim_offset=sc.get('aim_offset', (0, 0, 0.15)))


def cached_object_goals(exp, env, n):
    key = hashlib.sha1(f'{exp.scene.signature()}|{len(exp.pool)}|{n}|{env.aim_offset}|{env.target_object}'
                       .encode()).hexdigest()[:12]
    path = cache_dir() / f'object-goals-{key}.npy'
    if path.exists():
        return np.load(path)
    goals = object_goals(exp, env, n, seed=EVAL_SEED + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, goals)
    return goals


def save(out_dir: Path, name: str, summary: dict, distances, goals, extra=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{name}.json').write_text(json.dumps({**summary, **(extra or {})}, indent=2))
    np.savez_compressed(out_dir / f'{name}.npz', distances=distances.astype(np.float32), goals=goals)


def evaluate_policy(exp, policy, backend, modes, n_goals, out_dir, label, extra=None, chain=False):
    results = {}
    if backend == 'kinematic':
        goals = exp.eval_goals()
        evaluator = KinematicEvaluator(exp.scene, exp.bins, exp.reward, exp.horizon, goals, n_envs=32,
                                       check_collision=exp.config['mdp']['collision'])
        for mode in ('home', 'chain'):
            summary, distances = evaluator.run(policy, mode)
            save(out_dir, f'kinematic-{mode}-{label}', summary, distances, goals, extra)
            results[mode] = summary
    else:
        goals = None
        for mode in modes:
            env = physics_env(exp, mode)
            goals = cached_object_goals(exp, env, n_goals) if goals is None else goals
            summary, distances = physics_run(env, policy, goals, chain=chain)
            tag = 'chain' if chain else 'home'
            save(out_dir, f'libero{mode}-{tag}-{label}', summary, distances, goals, extra)
            results[mode] = summary
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', type=Path, help='training run directory')
    source.add_argument('--config', type=Path, help='config for a reference controller')
    parser.add_argument('--checkpoints', nargs='*', default=['all'])
    parser.add_argument('--baseline', choices=['random', 'dls', 'dls_continuous'])
    parser.add_argument('--backend', choices=['kinematic', 'libero'], default='kinematic')
    parser.add_argument('--modes', nargs='*', default=['A', 'B'])
    parser.add_argument('--goals', type=int, default=64)
    parser.add_argument('--chain', action='store_true', help='simulator: chain goals without resets')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    args = parser.parse_args(argv)

    if args.run:
        config = override(load_config(args.run / 'config.yaml'), args.set)
        exp = make_experiment(config, keep_env=args.backend == 'libero')
        names = []
        for token in args.checkpoints:      # 'all' expands to every step_*.pt
            names += sorted(p.name for p in (args.run / 'checkpoints').glob('step_*.pt')) if token == 'all' else [token]
        for name in names:
            agent, state = load_agent(args.run / 'checkpoints' / name)
            results = evaluate_policy(exp, agent.act, args.backend, args.modes, args.goals, args.run / 'eval',
                                      Path(name).stem, {'step': state['step'], 'updates': state['updates']},
                                      args.chain)
            for mode, summary in results.items():
                print(f'{name} {args.backend} {mode}: median {summary["final_median"]*100:.2f} cm  '
                      f'p90 {summary["final_p90"]*100:.2f} cm  success@2cm {summary["success_2cm"]:.2f}', flush=True)
    else:
        if not args.baseline:
            parser.error('--config needs --baseline')
        config = override(load_config(args.config), args.set)
        exp = make_experiment(config, keep_env=args.backend == 'libero')
        policy = {'random': lambda: RandomPolicy(exp.scene.n_joints, exp.bins),
                  'dls': lambda: DLSPolicy(exp.scene, exp.bins, quantise=True),
                  'dls_continuous': lambda: DLSPolicy(exp.scene, exp.bins, quantise=False)}[args.baseline]()
        out = repo_root() / 'runs' / 'baselines' / config['name'] / args.baseline / 'eval'
        results = evaluate_policy(exp, policy, args.backend, args.modes, args.goals, out, args.baseline,
                                  chain=args.chain)
        for mode, summary in results.items():
            print(f'{args.baseline} {args.backend} {mode}: median {summary["final_median"]*100:.2f} cm  '
                  f'p90 {summary["final_p90"]*100:.2f} cm  success@2cm {summary["success_2cm"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
