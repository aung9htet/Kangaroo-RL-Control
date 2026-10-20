"""Q-learning on the factored action space (DecQN, Seyde et al., ICLR 2023) with a Double-Q target.

Q(s, a) = (1/J) sum_j U_j(s, a_j). Because Q is a sum of per-joint terms, its maximum over the
K^J joint actions is the sum of per-joint maxima, so greedy action selection and the Bellman max are
exact and cost J * K instead of K^J. One TD error is fitted on the joint action actually taken:

    y = r + gamma * Q_target(s', a*),   a*_j = argmax_k U_j(s', k; online network)
    L = Huber(y - Q(s, a)).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .network import DecQNetwork


def joint_q(utilities, actions):
    """Q(s, a) from utilities (B, J, K) and integer actions (B, J)."""
    return utilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1).mean(-1)


class DecQNAgent:
    kind = 'additive'

    def __init__(self, network: dict, gamma=0.95, lr=3e-4, tau=0.005, grad_clip=10.0, reward_bounds=(-2.01, 1.0),
                 double=True, huber_delta=1.0, device='cpu'):
        self.device = torch.device(device)
        self.network_config = dict(network)
        self.net = DecQNetwork(**network).to(self.device)
        self.target = copy.deepcopy(self.net).requires_grad_(False)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.tau, self.grad_clip = gamma, tau, grad_clip
        self.double, self.huber_delta = double, huber_delta
        # With |r| bounded, every return lies in [r_min, r_max] / (1 - gamma); targets are clipped to it.
        self.y_min = reward_bounds[0] / (1.0 - gamma)
        self.y_max = reward_bounds[1] / (1.0 - gamma)
        self.updates = 0

    def _tensor(self, array, dtype=torch.float32):
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    @torch.no_grad()
    def utilities(self, q, goal):
        return self.net(self._tensor(q), self._tensor(goal)).cpu().numpy()

    @torch.no_grad()
    def greedy_and_value(self, q, goal):
        utilities = self.net(self._tensor(q), self._tensor(goal))
        return utilities.argmax(-1).cpu().numpy(), utilities.max(-1).values.mean(-1).cpu().numpy()

    @torch.no_grad()
    def act(self, q, goal):
        """Greedy joint action: the per-joint argmax is the argmax over all K^J actions."""
        return self.net(self._tensor(q), self._tensor(goal)).argmax(-1).cpu().numpy()

    @torch.no_grad()
    def td_target(self, reward, q_next, goal, discount=None):
        """y = R + discount * Q_target(s_next, argmax_online); discount = gamma (1-step) or gamma^m (n-step)."""
        q_next, goal = self._tensor(q_next), self._tensor(goal)
        target_u = self.target(q_next, goal)
        chooser = self.net(q_next, goal) if self.double else target_u
        discount = self.gamma if discount is None else self._tensor(discount)
        y = self._tensor(reward) + discount * joint_q(target_u, chooser.argmax(-1))
        return y.clamp(self.y_min, self.y_max)

    def update(self, batch: dict) -> dict:
        q, goal = self._tensor(batch['q']), self._tensor(batch['goal'])
        actions = self._tensor(batch['action'], torch.int64)
        y = self.td_target(batch['reward'], batch['q_next'], batch['goal'], batch.get('discount'))
        value = joint_q(self.net(q, goal), actions)
        loss = F.huber_loss(value, y, delta=self.huber_delta)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimiser.step()
        self.polyak()
        self.updates += 1
        return {'loss': float(loss), 'q_mean': float(value.mean()), 'grad_norm': float(norm)}

    @torch.no_grad()
    def polyak(self):
        for parameter, target in zip(self.net.parameters(), self.target.parameters()):
            target.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)

    def state(self) -> dict:
        return {'kind': self.kind, 'network_config': self.network_config, 'net': self.net.state_dict(),
                'target': self.target.state_dict(), 'optimiser': self.optimiser.state_dict(),
                'updates': self.updates}

    def load(self, state: dict, weights_only=False):
        self.net.load_state_dict(state['net'])
        self.target.load_state_dict(state.get('target', state['net']))
        if not weights_only and 'optimiser' in state:
            self.optimiser.load_state_dict(state['optimiser'])
        self.updates = state.get('updates', 0)


class Explorer:
    """Per-joint epsilon-greedy with temporally extended random actions: when a joint explores, it
    repeats one random bin for 1..max_repeat steps, so random motion travels instead of jittering."""

    def __init__(self, n_envs, n_joints, n_bins, max_repeat=10, seed=0):
        self.n_bins, self.max_repeat = n_bins, max_repeat
        self.rng = np.random.default_rng(seed)
        self.remaining = np.zeros((n_envs, n_joints), dtype=np.int64)
        self.random_action = np.zeros((n_envs, n_joints), dtype=np.int64)

    def reset(self):
        self.remaining[:] = 0

    def __call__(self, greedy, epsilon):
        actions = np.array(greedy, dtype=np.int64, copy=True)
        ongoing = self.remaining > 0
        actions[ongoing] = self.random_action[ongoing]
        self.remaining[ongoing] -= 1
        start = ~ongoing & (self.rng.random(actions.shape) < epsilon)
        count = int(start.sum())
        if count:
            self.random_action[start] = self.rng.integers(self.n_bins, size=count)
            self.remaining[start] = self.rng.integers(1, self.max_repeat + 1, size=count) - 1
            actions[start] = self.random_action[start]
        return actions
