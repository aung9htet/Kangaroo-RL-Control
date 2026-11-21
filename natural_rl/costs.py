"""Unsafe-proposal cost accounting and the PID Lagrangian multiplier (Stooke, Achiam and Abbeel, ICML 2020).

cost per decision = collision attempt + pre-clip joint-limit attempt   (counted on PROPOSED motion, so rejected
or clipped commands are never hidden). The constraint bounds the EXPECTED episodic cost by a registered budget;
it is not a per-episode or zero-event guarantee, and raw counts plus the budget-violation frequency are reported
next to every constrained result. The multiplier update below is the PID rule of the paper applied to an
episodic cost estimate; combining it with SAC is an adaptation, not something either paper proposes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def unsafe_cost(collision_proposed, clip_proposed) -> np.ndarray:
    """Per-decision cost: number of unsafe proposal events (0, 1 or 2) for each environment."""
    return np.asarray(collision_proposed, dtype=np.float64) + np.asarray(clip_proposed, dtype=np.float64)


def preclip_violation(q, v_cmd, dt, lower, upper) -> np.ndarray:
    """True where q + dt v leaves the joint range for ANY joint, before clipping (a proposed limit violation)."""
    proposed = np.asarray(q) + dt * np.asarray(v_cmd)
    return ((proposed < lower) | (proposed > upper)).any(axis=-1)


@dataclass
class PIDLagrangian:
    """lambda_{k+1} = max(0, Kp * e_k + Ki * I_k + Kd * D_k), e_k = J_c,k - budget.

    I_{k+1} = max(0, I_k + e_k) is the anti-windup integral; D_k = max(0, J_c,k - J_c,k-1) reacts only to rising
    cost, as in the paper. J_c is an exponential moving average of observed episodic costs, so one episode cannot
    jump the multiplier. `update` is called with the mean episodic cost of the latest batch of episodes.
    """
    budget: float
    kp: float = 0.1
    ki: float = 0.01
    kd: float = 0.01
    lambda_max: float = 100.0
    ema_alpha: float = 0.05
    lam: float = 0.0
    integral: float = 0.0
    cost_ema: float | None = None
    previous_ema: float | None = None
    history: list = field(default_factory=list)

    @classmethod
    def from_yaml(cls, budgets_cfg: dict, budget_id: str) -> 'PIDLagrangian':
        pid = budgets_cfg['pid']
        return cls(budget=float(budgets_cfg['budgets'][budget_id]['limit']), kp=float(pid['kp']), ki=float(pid['ki']),
                   kd=float(pid['kd']), lambda_max=float(pid['lambda_max']), ema_alpha=float(pid['ema_alpha']),
                   lam=float(pid['lambda_init']))

    def update(self, episodic_cost: float) -> float:
        self.previous_ema = self.cost_ema
        self.cost_ema = (episodic_cost if self.cost_ema is None
                         else (1 - self.ema_alpha) * self.cost_ema + self.ema_alpha * episodic_cost)
        error = self.cost_ema - self.budget
        self.integral = max(0.0, self.integral + error)
        derivative = 0.0 if self.previous_ema is None else max(0.0, self.cost_ema - self.previous_ema)
        self.lam = float(np.clip(self.kp * error + self.ki * self.integral + self.kd * derivative, 0.0, self.lambda_max))
        self.history.append({'episodic_cost': float(episodic_cost), 'cost_ema': float(self.cost_ema),
                             'lambda': self.lam, 'integral': self.integral})
        return self.lam

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in ('budget', 'kp', 'ki', 'kd', 'lambda_max', 'ema_alpha', 'lam',
                                              'integral', 'cost_ema', 'previous_ema')}

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            setattr(self, key, value)


def budget_violation_frequency(episode_costs, budget: float) -> float:
    """Share of episodes whose own cost exceeds the budget (the budget bounds the mean, so this is reported separately)."""
    costs = np.asarray(episode_costs, dtype=np.float64)
    return float(np.mean(costs > budget)) if len(costs) else float('nan')
