"""PPO (MultiDiscrete and continuous), SAC, TD3, TQC and CrossQ under the benchmark contract.

Library implementations (stable-baselines3 2.7.1, sb3-contrib 2.7.1) are reused unmodified; the only adaptations are
the vectorised ReachBench environment, the goal-relabelling replay for the off-policy methods, and the checkpoint
callback. ppo_md and ppo_box share every hyperparameter, so their contrast isolates the action representation.
"""
from __future__ import annotations

from importlib import import_module

import numpy as np
import torch as th

from ..replay import GoalReplay
from .sb3_common import BenchVecEnv, CheckpointCallback, GoalReplayBuffer, SB3Policy

ALGOS = {
    'ppo_md': ('stable_baselines3', 'PPO', 'multidiscrete'),
    'ppo_box': ('stable_baselines3', 'PPO', 'box'),
    'sac': ('stable_baselines3', 'SAC', 'box'),
    'td3': ('stable_baselines3', 'TD3', 'box'),
    'tqc': ('sb3_contrib', 'TQC', 'box'),
    'crossq': ('sb3_contrib', 'CrossQ', 'box'),
}


def algo_class(model_id: str):
    module, name, _ = ALGOS[model_id]
    return getattr(import_module(module), name)


def _make_policy_class(model_id: str):
    kind = ALGOS[model_id][2]

    class _Policy(SB3Policy):
        def __init__(self, model):
            super().__init__(model, model_id, kind)

        @classmethod
        def load(cls, directory, spec):
            return cls(SB3Policy.load_model(algo_class(model_id), directory))

    _Policy.__name__ = _Policy.__qualname__ = f'{model_id}Policy'
    _Policy.action_kind = kind
    return _Policy


PPOMDPolicy, PPOBoxPolicy = _make_policy_class('ppo_md'), _make_policy_class('ppo_box')
SACPolicy, TD3Policy = _make_policy_class('sac'), _make_policy_class('td3')
TQCPolicy, CrossQPolicy = _make_policy_class('tqc'), _make_policy_class('crossq')
POLICY_CLASSES = {'ppo_md': PPOMDPolicy, 'ppo_box': PPOBoxPolicy, 'sac': SACPolicy, 'td3': TD3Policy, 'tqc': TQCPolicy,
                  'crossq': CrossQPolicy}


def _pick_device(requested: str) -> str:
    return 'cpu' if requested == 'auto' else requested


def build_model(ctx, venv, replay=None):
    """The SB3 model for ctx.model with the registered hyperparameters (also used by the resolved-config record)."""
    h, algo, model_id = ctx.hyper, algo_class(ctx.model), ctx.model
    device = _pick_device(ctx.device)
    if model_id in ('ppo_md', 'ppo_box'):
        return algo('MlpPolicy', venv, learning_rate=float(h['lr']), n_steps=int(h['n_steps']), batch_size=int(h['batch_size']),
                    n_epochs=int(h['n_epochs']), gamma=float(h['gamma']), gae_lambda=float(h['gae_lambda']),
                    clip_range=float(h['clip_range']), ent_coef=float(h['ent_coef']),
                    policy_kwargs=dict(net_arch=dict(pi=list(h['net_arch']['pi']), vf=list(h['net_arch']['vf']))),
                    seed=ctx.seed, device=device, verbose=0)
    common = dict(batch_size=int(h['batch']), gamma=float(h['gamma']), learning_starts=int(h['learning_starts']),
                  buffer_size=int(h['replay']), train_freq=(1, 'step'), gradient_steps=-1 if int(h['utd']) == 1 else int(h['utd']),
                  replay_buffer_class=GoalReplayBuffer, replay_buffer_kwargs=dict(goal_replay=replay), seed=ctx.seed,
                  device=device, verbose=0)
    if h.get('library_defaults'):                                       # CrossQ: its paper-faithful library defaults
        return algo('MlpPolicy', venv, **common)
    common.update(learning_rate=float(h['lr']), tau=float(h['tau']), policy_kwargs=dict(net_arch=list(h['net_arch'])))
    if model_id == 'td3':
        from stable_baselines3.common.noise import NormalActionNoise
        sigma = float(h['exploration_noise_sigma'])
        return algo('MlpPolicy', venv, policy_delay=int(h['policy_delay']), target_policy_noise=float(h['target_policy_noise']),
                    target_noise_clip=float(h['target_noise_clip']),
                    action_noise=NormalActionNoise(np.zeros(venv.action_space.shape), sigma * np.ones(venv.action_space.shape)),
                    **common)
    if model_id == 'tqc':
        common['policy_kwargs'].update(n_critics=int(h['n_critics']), n_quantiles=int(h['n_quantiles']))
        return algo('MlpPolicy', venv, top_quantiles_to_drop_per_net=int(h['top_quantiles_to_drop_per_net']),
                    ent_coef=h['ent_coef'], **common)
    return algo('MlpPolicy', venv, ent_coef=h['ent_coef'], **common)      # sac


def train(ctx) -> None:
    model_id, h = ctx.model, ctx.hyper
    kind = ALGOS[model_id][2]
    th.set_num_threads(ctx.threads)
    J = ctx.scene.n_joints
    venv = BenchVecEnv(ctx.make_env(ctx.n_envs), kind)
    replay = None
    if model_id not in ('ppo_md', 'ppo_box'):
        if int(h['learning_starts']) < ctx.n_envs * ctx.horizon:
            raise ValueError(f'learning_starts ({h["learning_starts"]}) must be at least n_envs * horizon '
                             f'({ctx.n_envs * ctx.horizon}) so that a complete episode exists before the first update')
        replay = GoalReplay(int(h['replay']), ctx.horizon, J, J, np.float32, ctx.profile, ctx.spec, ctx.scene.lower, ctx.scene.upper,
                            her_ratio=float(h['her_ratio']), n_step=int(h['n_step']), gamma=float(h['gamma']), seed=ctx.seed)
    model = build_model(ctx, venv, replay)
    policy_class = POLICY_CLASSES[model_id]
    callback = CheckpointCallback(ctx.clock, lambda m: policy_class(m), log_episodes=ctx.log_episodes, log_updates=ctx.log_updates,
                                  total_steps=ctx.total_steps)
    model.learn(total_timesteps=ctx.total_steps, callback=callback, progress_bar=False)
    ctx.clock.on_end(ctx.total_steps, lambda directory: policy_class(model).save(directory))
