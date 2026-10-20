"""Decoupled Q-network: one utility per joint and velocity bin, Q(s, a) = mean_j U_j(s, a_j)."""
from __future__ import annotations

import torch
from torch import nn

from ..encoding import encoded_size, positional_encoding_torch


class DecQNetwork(nn.Module):
    """phi(q) and phi(g) are computed inside the network from raw values, so the replay buffer stores
    only joint angles and positions (the encoding is part of the model, not of the data)."""

    def __init__(self, q_lower, q_upper, goal_lower, goal_upper, n_bins: int, joint_freqs=4, goal_freqs=4,
                 hidden=256, depth=2):
        super().__init__()
        self.register_buffer('q_lower', torch.as_tensor(q_lower, dtype=torch.float32))
        self.register_buffer('q_upper', torch.as_tensor(q_upper, dtype=torch.float32))
        self.register_buffer('goal_lower', torch.as_tensor(goal_lower, dtype=torch.float32))
        self.register_buffer('goal_upper', torch.as_tensor(goal_upper, dtype=torch.float32))
        self.n_joints = len(q_lower)
        self.n_bins = n_bins
        self.joint_freqs, self.goal_freqs = joint_freqs, goal_freqs
        width = encoded_size(self.n_joints, joint_freqs) + encoded_size(3, goal_freqs)
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            width = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(width, self.n_joints * n_bins)

    def features(self, q, goal):
        q = 2.0 * (q - self.q_lower) / (self.q_upper - self.q_lower) - 1.0
        goal = 2.0 * (goal - self.goal_lower) / (self.goal_upper - self.goal_lower) - 1.0
        return torch.cat([positional_encoding_torch(q, self.joint_freqs),
                          positional_encoding_torch(goal, self.goal_freqs)], dim=-1)

    def forward(self, q, goal):
        """Utilities U, shape (batch, n_joints, n_bins)."""
        return self.head(self.trunk(self.features(q, goal))).view(-1, self.n_joints, self.n_bins)


class SequentialQNetwork(DecQNetwork):
    """Autoregressive per-joint Q-values: Q_j(s, a_1..a_{j-1}, k) for joint j, bin k.

    The trunk is the DecQN trunk; one stage head, shared by the joints, takes the trunk features, the bins
    already chosen for the earlier joints (one-hot, zero for later joints) and the stage index."""

    def __init__(self, *args, head_hidden=256, **kwargs):
        super().__init__(*args, **kwargs)
        trunk_width = self.head.in_features
        j, k = self.n_joints, self.n_bins
        self.stage_head = nn.Sequential(nn.Linear(trunk_width + j * k + j, head_hidden), nn.ReLU(),
                                        nn.Linear(head_hidden, k))
        del self.head
        self.register_buffer('earlier', torch.tril(torch.ones(j, j), diagonal=-1))   # earlier[j, i] = i < j
        self.register_buffer('stage_ids', torch.eye(j))

    def trunk_features(self, q, goal):
        return self.trunk(self.features(q, goal))

    def stages(self, q, goal, actions, hidden=None):
        """Q_j(s, a_<j, .) for every joint j, given the joint action `actions` (teacher forcing): (B, J, K)."""
        hidden = self.trunk_features(q, goal) if hidden is None else hidden
        b, j, k = len(hidden), self.n_joints, self.n_bins
        onehot = torch.nn.functional.one_hot(actions, k).float()                  # (B, J, K)
        previous = torch.einsum('ji,bik->bjik', self.earlier, onehot).reshape(b, j, j * k)
        x = torch.cat([hidden[:, None].expand(b, j, hidden.shape[-1]), previous,
                       self.stage_ids[None].expand(b, j, j)], dim=-1)
        return self.stage_head(x)

    def stage(self, hidden, chosen, index):
        """Q_index(s, chosen_<index, .) for one joint: (B, K)."""
        b, j, k = len(hidden), self.n_joints, self.n_bins
        onehot = torch.nn.functional.one_hot(chosen, k).float() * self.earlier[index][None, :, None]
        x = torch.cat([hidden, onehot.reshape(b, j * k), self.stage_ids[index][None].expand(b, j)], dim=-1)
        return self.stage_head(x)

    def greedy(self, q, goal):
        """Sequential argmax: joint 1 first, each next joint given the earlier choices. Returns the actions
        and the value max_k Q_1(s, k) (= max over all joint actions when the stages are consistent)."""
        hidden = self.trunk_features(q, goal)
        chosen = torch.zeros(len(hidden), self.n_joints, dtype=torch.int64, device=hidden.device)
        value = None
        for index in range(self.n_joints):
            values = self.stage(hidden, chosen, index)
            if index == 0:
                value = values.max(-1).values
            chosen[:, index] = values.argmax(-1)
        return chosen, value

    def forward(self, q, goal):
        raise NotImplementedError('Use stages() or greedy(); a sequential network has no independent utilities')
