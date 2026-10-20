"""Evaluation of a policy (q, g) -> bin indices on fixed goal sets, in the model and in the simulator."""
from __future__ import annotations

import numpy as np

from .envs.kinematic import KinematicReachEnv
from .metrics import summarise


class KinematicEvaluator:
    """Runs a policy in the model MDP on a fixed goal set.

    mode 'home'  : every goal from the home pose.
    mode 'chain' : n parallel chains; each goal starts where the previous goal of its chain ended
                   (retargeting: the object reappears somewhere else and the arm continues).
    """

    def __init__(self, scene, bins, reward, horizon, goals, n_envs=32, check_collision=True, active=None,
                 limits=None, start=None):
        self.env = KinematicReachEnv(scene, n_envs, bins, reward, horizon, check_collision, 0.0, active, limits)
        self.goals = np.asarray(goals, dtype=np.float64)
        self.horizon = horizon
        self.dt = bins.dt
        self.start = scene.home[self.env.active] if start is None else np.asarray(start)

    def _roll(self, policy, distances, collided, rows):
        q, goal = self.env.observe()
        for t in range(self.horizon):
            step, _ = self.env.step(policy(q, goal))
            q = step.q_next
            distances[rows, t] = step.distance[:len(rows)]
            collided[rows, t] = step.collided[:len(rows)]

    def run(self, policy, mode='home'):
        n, total = self.env.n, len(self.goals)
        distances = np.zeros((total, self.horizon))
        collided = np.zeros((total, self.horizon), dtype=bool)
        if mode == 'home':
            for begin in range(0, total, n):
                rows = np.arange(begin, min(begin + n, total))
                goals = self.goals[np.r_[rows, np.full(n - len(rows), rows[-1])]]
                self.env.reset_to(np.tile(self.start, (n, 1)), goals)
                self._roll(policy, distances, collided, rows)
        elif mode == 'chain':
            if total % n:
                raise ValueError('chain mode needs a goal count divisible by the number of environments')
            self.env.reset_to(np.tile(self.start, (n, 1)), self.goals[:n])
            for k in range(total // n):
                rows = np.arange(k * n, (k + 1) * n)
                self.env.set_goals(self.goals[rows])
                self._roll(policy, distances, collided, rows)
        else:
            raise ValueError(f'Unknown mode {mode!r}')
        return summarise(distances, self.dt, collided), distances


def object_goals(exp, physics_env, n, seed, margin=0.08, tolerance=0.01, max_candidates=600):
    """Goals for an object that appears on the table: the aim point above an object resting fully on
    the table top, kept only if the continuous Jacobian controller reaches it in the model (feasible)."""
    from .baselines import DLSPolicy

    scene = exp.scene
    height = physics_env.object_rest_height() + physics_env.aim_offset[2]
    x0, x1, y0, y1, _ = scene.table_box_base()
    points = exp.pool.points
    keep = ((np.abs(points[:, 2] - height) <= exp.pool_voxel) & (points[:, 0] >= x0 + margin)
            & (points[:, 1] >= y0 + margin) & (points[:, 1] <= y1 - margin))
    candidates = points[keep].copy()
    candidates[:, 2] = height
    rng = np.random.default_rng(seed)
    if len(candidates) > max_candidates:
        candidates = candidates[rng.choice(len(candidates), max_candidates, replace=False)]
    oracle = KinematicEvaluator(scene, exp.bins, exp.reward, exp.horizon, candidates, n_envs=32,
                                check_collision=exp.config['mdp']['collision'])
    _, distances = oracle.run(DLSPolicy(scene, exp.bins, quantise=False), 'home')
    feasible = candidates[distances[:, -1] <= tolerance]
    if len(feasible) < n:
        raise RuntimeError(f'Only {len(feasible)} feasible object goals; need {n}')
    return feasible[rng.choice(len(feasible), n, replace=False)]


def physics_run(env, policy, goals, chain=False, on_tick=None, on_goal=None):
    """Run the policy in the simulator on each goal (from home, or chained without resets)."""
    total, horizon = len(goals), env.horizon
    distances = np.zeros((total, horizon))
    collided = np.zeros((total, horizon), dtype=bool)
    tracking = np.zeros((total, horizon))
    shift = np.zeros(total)
    for i, goal in enumerate(goals):
        q, g = env.retarget(goal) if (chain and i > 0) else env.reset(goal)
        if on_goal is not None:
            on_goal(i, env)
        for t in range(horizon):
            step, _ = env.step(policy(q[None], g[None])[0], on_tick)
            q = step.q_next
            distances[i, t], collided[i, t], tracking[i, t] = step.distance, step.collided, step.tracking
        shift[i] = step.object_shift
    summary = summarise(distances, env.bins.dt, collided)
    summary.update(tracking_median=float(np.median(tracking)), tracking_p90=float(np.percentile(tracking, 90)),
                   object_shift_median=float(np.median(shift)), object_shift_max=float(shift.max()))
    return summary, distances
