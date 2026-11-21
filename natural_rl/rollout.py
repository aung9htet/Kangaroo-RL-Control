"""Deterministic episode rollouts shared by evaluation, `play` and the checkpoint videos."""
from __future__ import annotations

import numpy as np

from .models.base import Policy, to_velocity


def rollout(policy: Policy, env, task_index, seed: int | None = None, renderer=None) -> dict:
    """Run one full episode for each task in `task_index` (len == env.n) with the policy acting deterministically.

    Returns arrays with N = horizon + 1 samples for positions (sample 0 is the initial state) and `horizon`
    decisions for everything else. If `renderer` is given (single-environment only), the raw RGB frame of the
    accepted state is rendered at t = 0 and after every decision, rejected and still frames included.
    """
    if renderer is not None and env.n != 1:
        raise ValueError('video rendering uses a single environment')
    policy.reset()
    if seed is not None:
        policy.seed(seed)
    obs = env.reset(task_index)
    n, horizon = env.n, env.horizon
    ee, q = [env.p.copy()], [env.q.copy()]
    out = {k: [] for k in ('action', 'v_cmd', 'v_exec', 'collision_proposed', 'collision_rejected', 'clip_proposed', 'cost',
                           'reward', 'count', 'held_once')}
    raw = {k: [] for k in ('r0', 'hold', 'time', 'path', 'speed', 'accel', 'rom')}
    weighted = {k: [] for k in env.profile.terms}
    info_log = []
    frames = [renderer.render(env.q[0])] if renderer is not None else None
    for _ in range(horizon):
        action = policy.act(obs, deterministic=True)
        v = to_velocity(policy, action, env.obs_spec.v_max)
        obs, reward, done, info = env.step(v)
        ee.append(env.p.copy())
        q.append(env.q.copy())
        out['action'].append(np.asarray(action))
        out['reward'].append(reward)
        for key in ('v_cmd', 'v_exec', 'collision_proposed', 'collision_rejected', 'clip_proposed', 'cost', 'count', 'held_once'):
            out[key].append(info[key])
        for key in raw:
            raw[key].append(info['raw'][key])
        for key in weighted:
            weighted[key].append(info['weighted'][key])
        info_log.append(policy.info())
        if frames is not None:
            frames.append(renderer.render(env.q[0]))
    result = {k: np.stack(v, axis=1) for k, v in out.items()}
    result.update(ee=np.stack(ee, axis=1), q=np.stack(q, axis=1), goal=env.goal.copy(), task_index=np.asarray(task_index),
                  raw={k: np.stack(v, axis=1) for k, v in raw.items()},
                  weighted={k: np.stack(v, axis=1) for k, v in weighted.items()}, policy_info=info_log, dt=env.dt)
    if frames is not None:
        result['frames'] = np.stack(frames)
    return result
