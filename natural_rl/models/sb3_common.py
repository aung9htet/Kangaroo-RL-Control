"""Shared adapters for the Stable-Baselines3 / SB3-Contrib controllers (PPO, SAC, TD3, TQC, CrossQ, SAC-PID-Lagrangian).

* `BenchVecEnv`      : ReachBench as an SB3 VecEnv (auto-reset, time-limit truncation, per-step transition info);
* `GoalReplayBuffer` : an SB3 ReplayBuffer whose storage and sampling are the history-aware goal-relabelling
                       GoalReplay (relabelled goals, recomputed rewards/observations, n-step discounts, costs);
* `SB3Policy`        : the uniform Policy interface over a saved SB3 model;
* `CheckpointCallback`: step-0 / cadence / final / manual checkpoints (and their videos) driven by the shared Clock.
"""
from __future__ import annotations

import io
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv

from ..env.contract import N_BINS
from ..replay import GoalReplay
from .base import Policy, to_velocity


class BenchVecEnv(VecEnv):
    """n synchronous ReachBench environments. The episode ends at the horizon as a TRUNCATION (never a terminal)."""

    def __init__(self, env, action_kind: str):
        J = env.n_joints
        space = (gym.spaces.Box(-1.0, 1.0, (J,), dtype=np.float32) if action_kind == 'box'
                 else gym.spaces.MultiDiscrete([N_BINS] * J))
        super().__init__(env.n, gym.spaces.Box(-np.inf, np.inf, (env.obs_spec.dim,), dtype=np.float32), space)
        self.env, self.action_kind, self._actions = env, action_kind, None
        self._obs = env.reset()

    def reset(self):
        self._obs = self.env.reset()
        return self._obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        env = self.env
        v = to_velocity(self.action_kind, self._actions, env.obs_spec.v_max)
        obs, reward, done, info = env.step(v)
        shared = {'q': info['q'], 'q_next': info['q_next'], 'p': info['p'], 'p_next': info['p_next'], 'v_cmd': v,
                  'collided': info['collision_proposed'], 'clip': info['clip_proposed'], 'cost': info['cost'], 'qd_next': info['qd_next'],
                  'goal': env.goal.copy(), 't': env.t}
        infos = [{} for _ in range(self.num_envs)]
        infos[0]['batch_transition'] = shared
        infos[0]['cost_batch'] = info['cost']
        if done:
            ep = info['episode']
            for i in range(self.num_envs):
                infos[i]['terminal_observation'] = obs[i].copy()
                infos[i]['TimeLimit.truncated'] = True
                infos[i]['bench_episode'] = {'return': float(ep['return'][i]), 'cost': float(ep['cost'][i]),
                                             'collisions': float(ep['collisions'][i]), 'clips': float(ep['clips'][i]),
                                             'final_distance': float(ep['final_distance'][i]), 'task_index': int(ep['task_index'][i]),
                                             'terms': {k: float(v_[i]) for k, v_ in ep['terms'].items()}}
            obs = env.reset()
        self._obs = obs
        return obs, reward.astype(np.float32), np.full(self.num_envs, done), infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def seed(self, seed=None):
        return [seed] * self.num_envs


class GoalReplayBuffer(ReplayBuffer):
    """SB3 replay buffer API over GoalReplay. Constructed by SB3 via `replay_buffer_class` / `replay_buffer_kwargs`."""

    def __init__(self, buffer_size, observation_space, action_space, device='auto', n_envs=1, optimize_memory_usage=False,
                 handle_timeout_termination=False, goal_replay: GoalReplay | None = None, n_steps=1, gamma=0.99, **_):
        if goal_replay is None:
            raise ValueError('GoalReplayBuffer needs goal_replay=GoalReplay(...) in replay_buffer_kwargs')
        # BaseBuffer bookkeeping only: the real storage lives in GoalReplay, so keep the SB3 arrays minimal.
        super().__init__(1, observation_space, action_space, device, n_envs, False, False)
        self.replay = goal_replay
        self.t = 0
        self.buffer_size = max(1, goal_replay.episodes * goal_replay.horizon // max(1, n_envs))
        self.handle_timeout_termination = False

    # SB3 reads these in places; make them reflect the real content
    @property
    def pos(self):
        return int(self.replay.cursor)

    @pos.setter
    def pos(self, value):
        pass

    @property
    def full(self):
        return bool(self.replay.complete.all())

    @full.setter
    def full(self, value):
        pass

    def size(self) -> int:
        return len(self.replay)

    def add(self, obs, next_obs, action, reward, done, infos):
        batch = infos[0]['batch_transition']
        if self.t == 0:
            self.replay.begin(batch['goal'])
        self.replay.add(self.t, batch['q'], batch['q_next'], batch['p'], batch['p_next'], batch['v_cmd'],
                        np.asarray(action).reshape(len(batch['q']), -1), batch['collided'], batch['clip'], batch['qd_next'])
        self.t += 1
        if self.t == self.replay.horizon:
            self.replay.end()
            self.t = 0

    def sample(self, batch_size, env=None):
        s = self.replay.sample(batch_size)
        t = lambda a, shape=None: th.as_tensor(a if shape is None else np.asarray(a).reshape(shape), dtype=th.float32, device=self.device)
        samples = ReplayBufferSamples(observations=t(s['obs']), actions=t(s['action']), next_observations=t(s['next_obs']),
                                      dones=th.zeros((batch_size, 1), device=self.device), rewards=t(s['reward'], (-1, 1)),
                                      discounts=t(s['discount'], (-1, 1)))
        self.last_costs = th.as_tensor(s['cost'], dtype=th.float32, device=self.device).reshape(-1, 1)   # for cost critics
        return samples


class SB3Policy(Policy):
    """A saved SB3 / SB3-Contrib model acting deterministically through the uniform Policy interface."""

    def __init__(self, model: BaseAlgorithm, model_id: str, action_kind: str):
        self.model, self.model_id, self.action_kind = model, model_id, action_kind

    def act(self, obs, deterministic: bool = True):
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return action.astype(np.int64) if self.action_kind == 'multidiscrete' else action.astype(np.float32)

    def seed(self, seed: int) -> None:
        th.manual_seed(seed)

    def save(self, directory: Path) -> None:
        self.model.save(Path(directory) / 'model.zip', exclude=['replay_buffer'])

    @classmethod
    def load_model(cls, algo_cls, directory: Path):
        return algo_cls.load(Path(directory) / 'model.zip', device='cpu', env=None)


class CheckpointCallback(BaseCallback):
    """Drives the shared Clock from SB3's step loop: step 0 at the start, cadence/final/manual as steps accumulate."""

    def __init__(self, clock, make_policy, log_episodes=None, log_updates=None, log_every: int = 10_000, on_episode=None,
                 total_steps=None):
        super().__init__()
        self.on_episode, self.total_steps = on_episode, total_steps
        self.clock, self.make_policy, self.log_episodes = clock, make_policy, log_episodes
        self.log_updates, self.log_every, self._next_log = log_updates, log_every, 0

    def _save_fn(self):
        return lambda directory: self.make_policy(self.model).save(directory)

    def _on_training_start(self) -> None:
        self.clock.on_start(0, self._save_fn())

    def _on_step(self) -> bool:
        if self.log_episodes is not None:
            episodes = [i['bench_episode'] for i in self.locals.get('infos', []) if 'bench_episode' in i]
            if episodes and self.on_episode is not None:
                self.on_episode(self.num_timesteps, {'cost': np.array([e['cost'] for e in episodes])})
            if episodes:                                    # all synchronous environments finish together: array form
                self.log_episodes(self.num_timesteps, {
                    'return': np.array([e['return'] for e in episodes]), 'cost': np.array([e['cost'] for e in episodes]),
                    'collisions': np.array([e['collisions'] for e in episodes]), 'clips': np.array([e['clips'] for e in episodes]),
                    'final_distance': np.array([e['final_distance'] for e in episodes]),
                    'task_index': np.array([e['task_index'] for e in episodes]),
                    'terms': {k: np.array([e['terms'][k] for e in episodes]) for k in episodes[0]['terms']}})
        if self.log_updates is not None and self.num_timesteps >= self._next_log:
            values = {k.replace('/', '_'): float(v) for k, v in self.model.logger.name_to_value.items() if k.startswith('train/')}
            if values:
                self.log_updates(self.num_timesteps, int(getattr(self.model, '_n_updates', 0)), values)
                self._next_log = self.num_timesteps + self.log_every
        self.clock.after_step(self.num_timesteps, self._save_fn())
        # stop exactly at the registered budget (PPO would otherwise finish its whole rollout past the final checkpoint)
        return self.total_steps is None or self.num_timesteps < self.total_steps
