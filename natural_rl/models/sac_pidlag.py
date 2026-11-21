"""SAC with PID-Lagrangian expected-cost budgets: an ADAPTATION of two published methods.

SAC (Haarnoja et al., ICML 2018) is extended with a cost critic of the same architecture trained on the discounted
unsafe-proposal cost, and the actor maximises the normalised Lagrangian (Q_r - lambda Q_c) / (1 + lambda) as in the
PID Lagrangian methods of Stooke, Achiam and Abbeel (ICML 2020). The multiplier lambda is updated by their PID rule
from the measured undiscounted EPISODIC cost against a registered budget (`natural_rl.costs.PIDLagrangian`). The
combination is not attributed to either paper; the budget bounds the EXPECTED episodic cost and is not a per-episode
guarantee. Cost = collision attempts + pre-clip joint-limit attempts per decision.
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from .sb3_common import CheckpointCallback, GoalReplayBuffer, SB3Policy


class SACPIDLag(SAC):
    def __init__(self, *args, pid=None, **kwargs):
        self.pid = pid
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # Cost critics (twin, same architecture as the reward critics) live on the policy so they are saved with it.
        self.policy.cost_critic = self.policy.make_critic(features_extractor=None).to(self.device)
        self.policy.cost_critic_target = self.policy.make_critic(features_extractor=None).to(self.device)
        self.policy.cost_critic_target.load_state_dict(self.policy.cost_critic.state_dict())
        self.policy.cost_critic_target.set_training_mode(False)
        self.cost_critic, self.cost_critic_target = self.policy.cost_critic, self.policy.cost_critic_target
        self.cost_critic.optimizer = th.optim.Adam(self.cost_critic.parameters(), lr=self.lr_schedule(1))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.cost_critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        lam = 0.0 if self.pid is None else float(self.pid.lam)
        ent_coef_losses, ent_coefs, actor_losses, critic_losses, cost_losses = [], [], [], [], []

        for gradient_step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            costs = self.replay_buffer.last_costs
            discounts = data.discounts if data.discounts is not None else self.gamma
            actions_pi, log_prob = self.actor.action_log_prob(data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_q = th.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                next_q, _ = th.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = data.rewards + (1 - data.dones) * discounts * next_q
                next_qc = th.cat(self.cost_critic_target(data.next_observations, next_actions), dim=1).mean(dim=1, keepdim=True)
                target_qc = costs + (1 - data.dones) * discounts * next_qc          # no entropy term for the cost

            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in self.critic(data.observations, data.actions))
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            cost_loss = 0.5 * sum(F.mse_loss(q, target_qc) for q in self.cost_critic(data.observations, data.actions))
            cost_losses.append(cost_loss.item())
            self.cost_critic.optimizer.zero_grad()
            cost_loss.backward()
            self.cost_critic.optimizer.step()

            q_pi = th.min(th.cat(self.critic(data.observations, actions_pi), dim=1), dim=1, keepdim=True)[0]
            qc_pi = th.cat(self.cost_critic(data.observations, actions_pi), dim=1).mean(dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - (q_pi - lam * qc_pi) / (1.0 + lam)).mean()
            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.cost_critic.parameters(), self.cost_critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record('train/ent_coef', np.mean(ent_coefs))
        self.logger.record('train/actor_loss', np.mean(actor_losses))
        self.logger.record('train/critic_loss', np.mean(critic_losses))
        self.logger.record('train/cost_critic_loss', np.mean(cost_losses))
        self.logger.record('train/lambda_pid', lam)
        if ent_coef_losses:
            self.logger.record('train/ent_coef_loss', np.mean(ent_coef_losses))


class SACPIDLagPolicy(SB3Policy):
    action_kind = 'box'
    model_id = 'sac_pidlag'

    def __init__(self, model):
        super().__init__(model, 'sac_pidlag', 'box')

    @classmethod
    def load(cls, directory, spec):
        return cls(SB3Policy.load_model(SACPIDLag, directory))


def train(ctx) -> None:
    import torch
    from ..replay import GoalReplay
    from .sb3_common import BenchVecEnv
    from .sb3_models import _pick_device
    if ctx.cost_budget is None:
        raise ValueError('sac_pidlag needs --cost-budget-id (budget_moderate_v1 or budget_strict_v1)')
    h, pid, J = ctx.hyper, ctx.cost_budget, ctx.scene.n_joints
    torch.set_num_threads(ctx.threads)
    if int(h['learning_starts']) < ctx.n_envs * ctx.horizon:
        raise ValueError('learning_starts must be at least n_envs * horizon')
    venv = BenchVecEnv(ctx.make_env(ctx.n_envs), 'box')
    replay = GoalReplay(int(h['replay']), ctx.horizon, J, J, np.float32, ctx.profile, ctx.spec, ctx.scene.lower, ctx.scene.upper,
                        her_ratio=float(h['her_ratio']), n_step=int(h['n_step']), gamma=float(h['gamma']), seed=ctx.seed)
    model = SACPIDLag('MlpPolicy', venv, learning_rate=float(h['lr']), tau=float(h['tau']), gamma=float(h['gamma']),
                      batch_size=int(h['batch']), learning_starts=int(h['learning_starts']), buffer_size=int(h['replay']),
                      train_freq=(1, 'step'), gradient_steps=-1, ent_coef=h['ent_coef'],
                      policy_kwargs=dict(net_arch=list(h['net_arch'])), replay_buffer_class=GoalReplayBuffer,
                      replay_buffer_kwargs=dict(goal_replay=replay), seed=ctx.seed, device=_pick_device(ctx.device), verbose=0, pid=pid)

    def on_episode(step, episode):                          # mean undiscounted episodic cost of the finished block
        lam = pid.update(float(np.mean(episode['cost'])))
        ctx.log_updates(step, int(getattr(model, '_n_updates', 0)),
                        {'lambda_pid': lam, 'cost_ema': pid.cost_ema, 'episodic_cost_mean': float(np.mean(episode['cost'])),
                         'budget': pid.budget, 'budget_violation_frequency': float(np.mean(np.asarray(episode['cost']) > pid.budget))})

    def make_policy(m):
        return SACPIDLagPolicy(m)

    callback = CheckpointCallback(ctx.clock, make_policy, log_episodes=ctx.log_episodes, log_updates=ctx.log_updates, on_episode=on_episode,
                                  total_steps=ctx.total_steps)
    model.learn(total_timesteps=ctx.total_steps, callback=callback, progress_bar=False)
    ctx.clock.on_end(ctx.total_steps, lambda directory: make_policy(model).save(directory))
