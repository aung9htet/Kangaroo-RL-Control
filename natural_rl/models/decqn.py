"""DecQN for the benchmark (Seyde et al., ICLR 2023): the historical agent's mathematics on the flat observation.

Q(s, a) = (1/J) sum_j U_j(s, a_j); greedy actions are per-joint argmaxes (exact over all 7^7 joint actions);
Double-Q target with an n-step return from the goal-relabelling replay, Huber loss, Polyak target network,
gradient clipping, and targets clipped to the profile's reward bounds / (1 - gamma). `kangaroo_rl.agent.decqn`
stays the frozen historical implementation (used unchanged for the P0 parity runs); this class differs only in that
the sin/cos encoding is a fixed feature map applied by the environment (so every algorithm sees the same inputs)
instead of being computed inside the network. tests/bench/test_decqn.py proves the two agree on the shared inputs.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from kangaroo_rl.agent.decqn import Explorer, joint_q

from ..env.contract import N_BINS, multidiscrete_to_velocity
from ..replay import GoalReplay
from .base import Policy

# Growing Q-Networks (Seyde et al., L4DC 2024): nested active sub-spaces of the 7 historical levels
# {-1,-0.3,-0.1,0,0.1,0.3,1} -> bins 2 -> 3 -> 5 -> 7 (indices into the level list).
STAGES = ([0, 6], [0, 3, 6], [0, 1, 3, 5, 6], [0, 1, 2, 3, 4, 5, 6])


def stage_at(step: int, total: int, fractions) -> int:
    """Linear schedule: the number of growth boundaries (fractions of training) already passed."""
    return int(sum(step >= f * total for f in fractions))


class MaskedExplorer(Explorer):
    """Per-joint epsilon-greedy with repeated random bins, sampled only from the currently active bins."""

    def __init__(self, n_envs, n_joints, n_bins, max_repeat=10, seed=0):
        super().__init__(n_envs, n_joints, n_bins, max_repeat, seed)
        self.choices = np.arange(n_bins)

    def set_choices(self, indices):
        self.choices = np.asarray(list(indices))

    def __call__(self, greedy, epsilon):
        actions = np.array(greedy, dtype=np.int64, copy=True)
        ongoing = self.remaining > 0
        actions[ongoing] = self.random_action[ongoing]
        self.remaining[ongoing] -= 1
        start = ~ongoing & (self.rng.random(actions.shape) < epsilon)
        count = int(start.sum())
        if count:
            self.random_action[start] = self.choices[self.rng.integers(len(self.choices), size=count)]
            self.remaining[start] = self.rng.integers(1, self.max_repeat + 1, size=count) - 1
            actions[start] = self.random_action[start]
        return actions


class FlatDecQNetwork(nn.Module):
    """obs (B, D) -> utilities (B, J, K): trunk of `depth` x [Linear, LayerNorm, ReLU] and one linear head."""

    def __init__(self, obs_dim: int, n_joints: int = 7, n_bins: int = N_BINS, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.n_joints, self.n_bins = n_joints, n_bins
        layers, width = [], obs_dim
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_joints * n_bins)

    def forward(self, obs):
        return self.head(self.trunk(obs)).view(-1, self.n_joints, self.n_bins)


class BenchDecQNAgent:
    def __init__(self, obs_dim: int, reward_bounds, n_joints=7, hidden=256, depth=2, gamma=0.95, lr=3e-4, tau=0.005,
                 grad_clip=10.0, huber_delta=1.0, double=True, device='cpu'):
        self.device = torch.device(device)
        self.arch = dict(obs_dim=obs_dim, n_joints=n_joints, n_bins=N_BINS, hidden=hidden, depth=depth)
        self.net = FlatDecQNetwork(**self.arch).to(self.device)
        self.target = copy.deepcopy(self.net).requires_grad_(False)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.tau, self.grad_clip, self.double, self.huber_delta = gamma, tau, grad_clip, double, huber_delta
        self.y_min, self.y_max = reward_bounds[0] / (1.0 - gamma), reward_bounds[1] / (1.0 - gamma)
        self.updates = 0
        self.active = None                                  # bool mask over bins (Growing Q); None = all bins

    def set_active(self, indices):
        mask = torch.zeros(N_BINS, dtype=torch.bool, device=self.device)
        mask[list(indices)] = True
        self.active = mask

    def _masked(self, utilities):
        return utilities if self.active is None else utilities.masked_fill(~self.active, float('-inf'))

    def _t(self, array, dtype=torch.float32):
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    @torch.no_grad()
    def act(self, obs) -> np.ndarray:
        return self._masked(self.net(self._t(obs))).argmax(-1).cpu().numpy()

    @torch.no_grad()
    def td_target(self, reward, next_obs, discount):
        next_obs = self._t(next_obs)
        target_u = self.target(next_obs)
        chooser = self._masked(self.net(next_obs) if self.double else target_u)
        y = self._t(reward) + self._t(discount) * joint_q(target_u, chooser.argmax(-1))
        return y.clamp(self.y_min, self.y_max)

    def update(self, batch: dict) -> dict:
        obs, actions = self._t(batch['obs']), self._t(batch['action'], torch.int64)
        y = self.td_target(batch['reward'], batch['next_obs'], batch['discount'])
        value = joint_q(self.net(obs), actions)
        loss = F.huber_loss(value, y, delta=self.huber_delta)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimiser.step()
        with torch.no_grad():
            for p, tp in zip(self.net.parameters(), self.target.parameters()):
                tp.mul_(1.0 - self.tau).add_(p, alpha=self.tau)
        self.updates += 1
        return {'loss': float(loss), 'q_mean': float(value.mean()), 'grad_norm': float(norm)}


class DecQNPolicy(Policy):
    action_kind = 'multidiscrete'
    model_id = 'decqn'

    def __init__(self, net: FlatDecQNetwork, arch: dict, active=None):
        self.net, self.arch = net.eval(), arch
        self.active = None if active is None else [int(i) for i in active]

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True) -> np.ndarray:
        utilities = self.net(torch.as_tensor(np.asarray(obs), dtype=torch.float32))
        if self.active is not None:                          # Growing Q: act only within the saved active sub-space
            mask = torch.zeros(N_BINS, dtype=torch.bool)
            mask[self.active] = True
            utilities = utilities.masked_fill(~mask, float('-inf'))
        return utilities.argmax(-1).numpy()

    def info(self) -> dict:
        return {'active_bins': len(self.active) if self.active is not None else N_BINS}

    def save(self, directory: Path) -> None:
        torch.save({'arch': self.arch, 'net': self.net.state_dict(), 'active': self.active}, Path(directory) / 'policy.pt')

    @classmethod
    def load(cls, directory: Path, spec) -> 'DecQNPolicy':
        state = torch.load(Path(directory) / 'policy.pt', map_location='cpu', weights_only=True)
        net = FlatDecQNetwork(**state['arch'])
        net.load_state_dict(state['net'])
        return cls(net, state['arch'], state.get('active'))


def policy_from_agent(agent: BenchDecQNAgent) -> DecQNPolicy:
    """A detached CPU copy of the online network (training keeps running on the original)."""
    net = FlatDecQNetwork(**agent.arch)
    net.load_state_dict({k: v.detach().cpu() for k, v in agent.net.state_dict().items()})
    active = None if agent.active is None else torch.nonzero(agent.active).flatten().tolist()
    return DecQNPolicy(net, dict(agent.arch), active)


def epsilon(step: int, total: int, start=1.0, end=0.02, fraction=0.15) -> float:
    return start + min(1.0, step / max(1.0, fraction * total)) * (end - start)


def train(ctx, growing: bool = False) -> None:
    """Train DecQN (or Growing Q with `growing=True`) under the benchmark contract; `ctx` is natural_rl.train.TrainContext."""
    h = ctx.hyper
    n, J = ctx.n_envs, ctx.scene.n_joints
    torch.manual_seed(ctx.seed)
    torch.set_num_threads(ctx.threads)
    device = 'cpu' if ctx.device == 'auto' else ctx.device          # small networks: CPU is faster and reproducible
    env = ctx.make_env(n)
    agent = BenchDecQNAgent(ctx.spec.dim, ctx.profile.bounds(J), J, int(h['hidden']), int(h['depth']), float(h['gamma']),
                            float(h['lr']), float(h['tau']), float(h['grad_clip']), device=device)
    replay = GoalReplay(int(h['replay']), ctx.horizon, J, J, np.int8, ctx.profile, ctx.spec, ctx.scene.lower,
                        ctx.scene.upper, her_ratio=float(h['her_ratio']), n_step=int(h['n_step']), gamma=float(h['gamma']),
                        seed=ctx.seed)
    explorer = MaskedExplorer(n, J, N_BINS, int(h['max_repeat']), seed=ctx.seed)
    fractions, stage = list(h.get('growth_fractions', [])), -1
    warmup, per_step = int(h['warmup']), int(h['updates_per_step'])
    obs = env.reset()
    replay.begin(env.goal)
    if growing:                                              # the step-0 checkpoint already carries the coarsest sub-space
        stage = 0
        agent.set_active(STAGES[0])
        explorer.set_choices(STAGES[0])
    ctx.clock.on_start(0, lambda directory: policy_from_agent(agent).save(directory))
    step, losses = 0, []
    while step < ctx.total_steps:
        if growing and stage_at(step, ctx.total_steps, fractions) != stage:
            stage = stage_at(step, ctx.total_steps, fractions)
            agent.set_active(STAGES[stage])
            explorer.set_choices(STAGES[stage])
            ctx.log_updates(step, agent.updates, {'stage': stage, 'active_bins': len(STAGES[stage])})
        eps = epsilon(step, ctx.total_steps, float(h['eps_start']), float(h['eps_end']), float(h['eps_fraction']))
        greedy = agent.act(obs) if step >= warmup else np.zeros((n, J), dtype=np.int64)
        actions = explorer(greedy, eps if step >= warmup else 1.0)
        v = multidiscrete_to_velocity(actions, ctx.spec.v_max)
        obs, _, done, info = env.step(v)
        replay.add(env.t - 1, info['q'], info['q_next'], info['p'], info['p_next'], v, actions,
                   info['collision_proposed'], info['clip_proposed'], info['qd_next'])
        step += n
        if done:
            replay.end()
            ctx.log_episodes(step, info['episode'])
            obs = env.reset()
            replay.begin(env.goal)
            explorer.reset()
        if step >= warmup and len(replay) > 0:
            for _ in range(per_step):
                losses.append(agent.update(replay.sample(int(h['batch']))))
            if agent.updates % 1000 < per_step or agent.updates == per_step:        # every 1000 updates, and the first batch
                ctx.log_updates(step, agent.updates, {'epsilon': eps, **{k: float(np.mean([x[k] for x in losses])) for k in losses[0]}})
                losses = []
        ctx.clock.after_step(step, lambda directory: policy_from_agent(agent).save(directory))
    if losses:                                                                              # a final row for short runs
        ctx.log_updates(step, agent.updates, {'epsilon': eps, **{k: float(np.mean([x[k] for x in losses])) for k in losses[0]}})
    ctx.clock.on_end(step, lambda directory: policy_from_agent(agent).save(directory))
