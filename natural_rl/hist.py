"""P0 historical DecQN parity runs (`--contract hist_v1`): the frozen `kangaroo_rl` training loop, unchanged.

The only differences from the original `python -m kangaroo_rl.train --config configs/reach.yaml` are the checkpoint
schedule (step 0, every 50,000 transitions, final) and a callback that converts each checkpoint to the benchmark policy
format so its raw video and evaluation can be produced. The resolved config must equal configs/reach.yaml except for
`train.checkpoints` (and any explicit `--set`, which marks the run as a non-parity smoke run). The original native outputs
(`config.yaml`, `run.json`, `metrics.csv`, `train.csv`, `checkpoints/*.pt`) are kept under <run>/native/.
"""
from __future__ import annotations

from .checkpoints import scheduled_steps
from .models.decqn import FlatDecQNetwork, DecQNPolicy
from .env.contract import N_BINS


def hist_policy_from_agent(agent) -> DecQNPolicy:
    """The historical DecQNAgent's network as a benchmark policy on obs_v1 (identical function, see tests/bench/test_decqn.py)."""
    net_cfg = agent.network_config
    width = 63 + 27 if net_cfg['joint_freqs'] == 4 and net_cfg['goal_freqs'] == 4 else None
    if width is None:
        raise ValueError('hist_v1 expects the historical sin/cos L = 4 encoding')
    arch = dict(obs_dim=width, n_joints=len(net_cfg['q_lower']), n_bins=N_BINS, hidden=int(net_cfg['hidden']), depth=int(net_cfg['depth']))
    net = FlatDecQNetwork(**arch)
    for name in ('trunk', 'head'):
        getattr(net, name).load_state_dict({k: v.detach().cpu() for k, v in getattr(agent.net, name).state_dict().items()})
    return DecQNPolicy(net, arch)


def parity_differences(resolved: dict, reference: dict, prefix: str = '') -> list:
    """Dotted keys at which two configs differ (used to prove the P0 config equals configs/reach.yaml)."""
    keys, diffs = set(resolved) | set(reference), []
    for key in sorted(keys):
        a, b = resolved.get(key), reference.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            diffs += parity_differences(a, b, f'{prefix}{key}.')
        elif a != b:
            diffs.append(f'{prefix}{key}')
    return diffs


def train(ctx, args, overrides) -> list:
    """Run the frozen historical loop for `ctx`; returns the dotted config keys that differ from configs/reach.yaml."""
    from kangaroo_rl.config import load_config, override
    from kangaroo_rl.train import train as historical_train

    from .registry import repo_root
    reference = load_config(repo_root() / 'configs' / 'reach.yaml')
    config = override(reference, list(overrides) + [f'train.steps={ctx.total_steps}'])
    steps = scheduled_steps(ctx.total_steps, ctx.checkpoint_every)

    def on_checkpoint(step, agent):
        kind = 'initial' if step == 0 else ('final' if step == ctx.total_steps else 'scheduled')
        ctx.clock._save(step, kind, lambda directory: hist_policy_from_agent(agent).save(directory))

    differences = [d for d in parity_differences(config, reference) if d not in ('train.steps',)]
    historical_train(config, ctx.seed, ctx.run_dir / 'native', 'cpu' if ctx.device == 'auto' else ctx.device, ctx.threads,
                     on_checkpoint=on_checkpoint, checkpoint_steps=steps)
    ctx.clock.on_end(ctx.total_steps, lambda directory: None)
    return differences
