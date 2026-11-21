"""TD-MPC2 (Hansen, Su and Wang, ICLR 2024) under the benchmark contract.

The official agent (vendored, see the docstring of natural_rl/vendor/tdmpc2/__init__.py) is used unchanged: latent world model, MPPI planning
at every decision, discrete-regression reward and value heads. Adaptations, all outside the algorithm:
  * replay: windows of horizon + 1 consecutive steps come from the goal-relabelling GoalReplay. Each window gets its own
    relabelled goal (probability her_ratio), and the dwell counter, rewards and observations are recomputed over the window;
  * environment: one environment (the planner handles one observation at a time), no terminal states (time-limit truncation);
  * evaluation: one planner state per environment, because evaluation runs many tasks in lockstep.
Replay of a saved checkpoint is deterministic given the same device and seed (even in eval mode the planner samples among elite
trajectories, so `policy.seed()` is part of the contract). The device used to load a policy is `NATURAL_RL_POLICY_DEVICE`, else
CUDA if available, else CPU; use the same one for recording and re-recording a video.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ..env.contract import box_to_velocity
from ..replay import GoalReplay
from ..vendor.tdmpc2 import TDMPC2
from ..vendor.tdmpc2.common import MODEL_SIZE
from .base import Policy


def default_device() -> str:
    return os.environ.get('NATURAL_RL_POLICY_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')


def build_cfg(hyper: dict, obs_dim: int, action_dim: int, horizon_steps: int, device: str):
    """Official config fields for a single-task, state-based, non-episodic problem (sizes from the official presets)."""
    h = dict(hyper)
    cfg = {k: h[k] for k in ('batch_size', 'reward_coef', 'value_coef', 'consistency_coef', 'rho', 'lr', 'enc_lr_scale',
                              'grad_clip_norm', 'tau', 'iterations', 'num_samples', 'num_elites', 'num_pi_trajs', 'horizon',
                              'min_std', 'max_std', 'temperature', 'log_std_min', 'log_std_max', 'entropy_coef', 'num_bins',
                              'vmin', 'vmax', 'num_channels', 'dropout', 'simnorm_dim', 'discount_denom', 'discount_min',
                              'discount_max')}
    cfg.update(MODEL_SIZE[int(h['model_size'])])
    cfg.setdefault('num_q', 5)
    cfg.update(task='reach', obs='state', episodic=False, multitask=False, mpc=True, compile=False, termination_coef=1,
               task_dim=0, tasks=['reach'], obs_shape={'state': [obs_dim]}, action_dim=action_dim, episode_length=horizon_steps,
               episode_lengths=[horizon_steps], action_dims=[action_dim], device=device,
               bin_size=(float(h['vmax']) - float(h['vmin'])) / int(h['num_bins']))
    return OmegaConf.create(cfg)


class SequenceBuffer:
    """What TD-MPC2's `agent.update(buffer)` calls: `.sample()` -> (obs, action, reward, terminated, task) as tensors.

    obs (h+1, B, D), action (h, B, A), reward (h, B, 1), terminated zeros (h, B, 1), task None. Windows start at step s in
    [0, H-1-h], so that the observation h steps later still precedes a decision of the same episode.
    """

    def __init__(self, replay: GoalReplay, horizon: int, batch: int, device: str):
        self.replay, self.h, self.batch, self.device = replay, horizon, batch, torch.device(device)

    def sample_arrays(self) -> dict:
        r, h, b = self.replay, self.h, self.batch
        complete = np.flatnonzero(r.complete)
        e = complete[r.rng.integers(len(complete), size=b)]
        s = r.rng.integers(r.horizon - h, size=b)                          # s in [0, H-1-h]
        goal = r.goal[e].astype(np.float64).copy()
        relabel = r.rng.random(b) < r.her_ratio
        future = s + (r.rng.random(b) * (r.horizon - s)).astype(np.int64)
        goal[relabel] = r.p_next[e[relabel], future[relabel]]
        hist = r._history(e, goal)                                          # dwell, first-hold and rewards for THIS goal
        reward, _ = r.profile.combine(hist)
        idx = np.arange(b)
        obs = np.stack([r.observations(e, s + k, goal, hist) for k in range(h + 1)])
        action = np.stack([r.action[e, s + k] for k in range(h)])
        rew = np.stack([reward[idx, s + k] for k in range(h)])[..., None]
        return {'obs': obs, 'action': action, 'reward': rew.astype(np.float32), 'episode': e, 'start': s, 'goal': goal}

    def sample(self):
        a = self.sample_arrays()
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return t(a['obs']), t(a['action']), t(a['reward']), torch.zeros_like(t(a['reward'])), None


class TDMPC2Policy(Policy):
    action_kind = 'box'
    model_id = 'tdmpc2'

    def __init__(self, agent: TDMPC2, cfg):
        self.agent, self.cfg = agent, cfg
        self.t, self.prev = 0, None

    def reset(self) -> None:
        self.t, self.prev = 0, None

    def seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self.prev is None:                                              # one planner state per environment
            self.prev = torch.zeros(len(obs), self.cfg.horizon, self.cfg.action_dim, device=self.agent.device)
        actions = []
        for i, row in enumerate(obs):
            self.agent._prev_mean.copy_(self.prev[i])
            a = self.agent.act(torch.as_tensor(row), t0=self.t == 0, eval_mode=deterministic)
            self.prev[i] = self.agent._prev_mean
            actions.append(a.numpy())
        self.t += 1
        return np.stack(actions).astype(np.float32)

    def save(self, directory: Path) -> None:
        torch.save({'cfg': OmegaConf.to_container(self.cfg), 'model': self.agent.model.state_dict(),
                    'scale': {k: v.cpu() for k, v in self.agent.scale.state_dict().items()}}, Path(directory) / 'policy.pt')

    @classmethod
    def load(cls, directory: Path, spec) -> 'TDMPC2Policy':
        state = torch.load(Path(directory) / 'policy.pt', map_location='cpu', weights_only=True)
        cfg = OmegaConf.create(state['cfg'])
        cfg.device = default_device()
        agent = TDMPC2(cfg)
        agent.model.load_state_dict(state['model'])
        agent.scale.load_state_dict({k: v.to(cfg.device) for k, v in state['scale'].items()})
        agent.model.eval()
        return cls(agent, cfg)


def train(ctx) -> None:
    h = ctx.hyper
    device = default_device() if ctx.device == 'auto' else ctx.device
    torch.manual_seed(ctx.seed)
    torch.set_num_threads(ctx.threads)
    J = ctx.scene.n_joints
    env = ctx.make_env(1)
    cfg = build_cfg(h, ctx.spec.dim, J, ctx.horizon, device)
    agent = TDMPC2(cfg)
    seed_steps = int(h['seed_steps'])
    if seed_steps % ctx.horizon or seed_steps <= 0:
        raise ValueError(f'seed_steps ({seed_steps}) must be a positive multiple of the horizon ({ctx.horizon})')
    replay = GoalReplay(int(h['replay']), ctx.horizon, J, J, np.float32, ctx.profile, ctx.spec, ctx.scene.lower, ctx.scene.upper,
                        her_ratio=float(h['her_ratio']), n_step=1, gamma=float(h['discount_max']), seed=ctx.seed)
    buffer = SequenceBuffer(replay, int(cfg.horizon), int(cfg.batch_size), device)
    policy = lambda: TDMPC2Policy(agent, cfg)
    rng = np.random.default_rng(ctx.seed)
    obs, first = env.reset(), True
    replay.begin(env.goal)
    ctx.clock.on_start(0, lambda directory: policy().save(directory))
    step, updates, stats = 0, 0, []
    while step < ctx.total_steps:
        if step < seed_steps:
            action = rng.uniform(-1, 1, (1, J)).astype(np.float32)
        else:
            action = agent.act(torch.as_tensor(obs[0], dtype=torch.float32), t0=first).numpy()[None]
        v = box_to_velocity(action, ctx.spec.v_max)
        obs, _, done, info = env.step(v)
        replay.add(env.t - 1, info['q'], info['q_next'], info['p'], info['p_next'], v, action, info['collision_proposed'],
                   info['clip_proposed'], info['qd_next'])
        step += 1
        first = False
        if done:
            replay.end()
            ctx.log_episodes(step, info['episode'])
            obs, first = env.reset(), True
            replay.begin(env.goal)
        if step >= seed_steps and len(replay) > 0:
            for _ in range(seed_steps if step == seed_steps else 1):       # official schedule: pretrain once, then 1 update/step
                info_u = agent.update(buffer)
                updates += 1
                if updates % 100 == 0:
                    stats.append({k: float(v_) for k, v_ in info_u.items()})
                if updates % 1000 == 0 or updates == 100:
                    ctx.log_updates(step, updates, {k: float(np.mean([s[k] for s in stats])) for k in stats[0]})
                    stats = []
        ctx.clock.after_step(step, lambda directory: policy().save(directory))
    ctx.clock.on_end(step, lambda directory: policy().save(directory))
