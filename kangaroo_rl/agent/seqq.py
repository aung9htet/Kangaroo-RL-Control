"""Q-learning with an autoregressive factorisation of the joint action (Metz et al. 2017; Chebotar et al.
2023, Q-Transformer), the alternative to the additive DecQN factorisation.

Joint j's values are conditioned on the bins chosen for joints 1..j-1. Each joint is a stage of a Bellman
backup inside one decision, and the last joint's stage backs up through time:

    Q_j(s, a_<j, a_j) = max_k Q_{j+1}(s, a_<=j, k)                      for j < J
    Q_J(s, a)         = R + gamma^m max_k Q_1(s', k)                    (n-step, Double-Q)

so max_k Q_1(s, k) = max over all K^J joint actions, and the sequential argmax is the exact greedy action.
Unlike the additive form it can represent how the joints interact, e.g. that small moves of all seven
joints add up to an overshoot near the goal.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .network import SequentialQNetwork


class SequentialQAgent:
    kind = 'sequential'

    def __init__(self, network: dict, gamma=0.95, lr=3e-4, tau=0.005, grad_clip=10.0, reward_bounds=(-2.01, 1.0),
                 double=True, huber_delta=1.0, device='cpu'):
        self.device = torch.device(device)
        self.network_config = dict(network)
        self.net = SequentialQNetwork(**network).to(self.device)
        self.target = copy.deepcopy(self.net).requires_grad_(False)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.tau, self.grad_clip = gamma, tau, grad_clip
        self.double, self.huber_delta = double, huber_delta
        self.y_min = reward_bounds[0] / (1.0 - gamma)
        self.y_max = reward_bounds[1] / (1.0 - gamma)
        self.updates = 0

    def _tensor(self, array, dtype=torch.float32):
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    @torch.no_grad()
    def act(self, q, goal):
        return self.net.greedy(self._tensor(q), self._tensor(goal))[0].cpu().numpy()

    @torch.no_grad()
    def greedy_and_value(self, q, goal):
        actions, value = self.net.greedy(self._tensor(q), self._tensor(goal))
        return actions.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def targets(self, batch):
        """Per-stage targets (B, J): within-step maxima for joints 1..J-1, the time backup for joint J."""
        q, goal = self._tensor(batch['q']), self._tensor(batch['goal'])
        actions = self._tensor(batch['action'], torch.int64)
        inner = self.target.stages(q, goal, actions)[:, 1:].max(-1).values            # (B, J-1)
        q_next = self._tensor(batch['q_next'])
        hidden_t = self.target.trunk_features(q_next, goal)
        zeros = torch.zeros(len(q_next), self.net.n_joints, dtype=torch.int64, device=self.device)
        first_t = self.target.stage(hidden_t, zeros, 0)
        if self.double:
            first_o = self.net.stage(self.net.trunk_features(q_next, goal), zeros, 0)
            bootstrap = first_t.gather(1, first_o.argmax(-1, keepdim=True)).squeeze(1)
        else:
            bootstrap = first_t.max(-1).values
        discount = self.gamma if batch.get('discount') is None else self._tensor(batch['discount'])
        last = self._tensor(batch['reward']) + discount * bootstrap
        return torch.cat([inner, last[:, None]], dim=1).clamp(self.y_min, self.y_max)

    def update(self, batch: dict) -> dict:
        q, goal = self._tensor(batch['q']), self._tensor(batch['goal'])
        actions = self._tensor(batch['action'], torch.int64)
        y = self.targets(batch)
        taken = self.net.stages(q, goal, actions).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.huber_loss(taken, y, delta=self.huber_delta)
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimiser.step()
        with torch.no_grad():
            for parameter, target in zip(self.net.parameters(), self.target.parameters()):
                target.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)
        self.updates += 1
        return {'loss': float(loss), 'q_mean': float(taken[:, -1].mean()), 'grad_norm': float(norm)}

    def state(self) -> dict:
        return {'kind': self.kind, 'network_config': self.network_config, 'net': self.net.state_dict(),
                'target': self.target.state_dict(), 'optimiser': self.optimiser.state_dict(), 'updates': self.updates}

    def load(self, state: dict, weights_only=False):
        self.net.load_state_dict(state['net'])
        self.target.load_state_dict(state.get('target', state['net']))
        if not weights_only and 'optimiser' in state:
            self.optimiser.load_state_dict(state['optimiser'])
        self.updates = state.get('updates', 0)
