"""Builds the pieces of an experiment from its config, so every entry point uses the same definitions."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mdp import GoalPool, GoalRegion, Reward, StartSampler, VelocityBins
from .scene import Scene, build_scene

EVAL_SEED = 12345     # the held-out goal set is the same for every run and seed


def repo_root() -> Path:
    return Path(os.environ.get('KANGAROO_RL_REPO', Path(__file__).resolve().parents[1]))


def cache_dir() -> Path:
    return repo_root() / '.runtime' / 'cache'


@dataclass
class Experiment:
    config: dict
    scene: Scene
    bins: VelocityBins
    reward: Reward
    region: GoalRegion
    pool: GoalPool

    @property
    def pool_voxel(self) -> float:
        return float(self.config['goals']['voxel'])

    @property
    def horizon(self) -> int:
        return int(self.config['mdp']['horizon'])

    def eval_goals(self, n=None):
        n = n or int(self.config['train']['eval_goals'])
        return self.pool.sample(np.random.default_rng(EVAL_SEED), n)

    @property
    def active(self):
        """Indices of the controlled joints (all 7 unless the config restricts them)."""
        active = self.config['mdp'].get('active')
        return np.arange(self.scene.n_joints) if active is None else np.asarray(active)

    @property
    def limits(self):
        """Joint limits of the active joints: the robot's, or the config's (2-joint lattice)."""
        limits = self.config['mdp'].get('limits')
        if limits is None:
            return np.stack([self.scene.lower[self.active], self.scene.upper[self.active]], axis=1)
        return np.asarray(limits, dtype=np.float64)

    def starts(self) -> StartSampler:
        return StartSampler(self.scene, self.config['starts']['home_noise'], self.active, self.limits)

    def network_config(self) -> dict:
        agent = self.config['agent']
        return dict(q_lower=self.limits[:, 0].tolist(), q_upper=self.limits[:, 1].tolist(),
                    goal_lower=self.pool.lower.tolist(), goal_upper=self.pool.upper.tolist(),
                    n_bins=self.bins.n_bins, joint_freqs=agent['joint_freqs'], goal_freqs=agent['goal_freqs'],
                    hidden=agent['hidden'], depth=agent['depth'])


def make_bins(config) -> VelocityBins:
    mdp = config['mdp']
    return VelocityBins(levels=tuple(float(v) for v in mdp['levels']), v_max=float(mdp['v_max']),
                        dt=float(mdp['dt']))


def make_reward(config, n_joints=7) -> Reward:
    r = config['reward']
    return Reward(d_far=r['d_far'], sigma=r['sigma'], lam_c=r['lam_c'], lam_a=r['lam_a'],
                  v_max=float(config['mdp']['v_max']), n_joints=n_joints, shaping=float(r.get('shaping', 0.0)),
                  gamma=float(config['agent']['gamma']))


def make_experiment(config, keep_env=False, scene=None) -> Experiment:
    sc = config['scene']
    if scene is None:
        scene = build_scene(suite=sc['suite'], task_id=sc['task_id'], seed=sc['seed'],
                            sim_overrides=sc.get('sim_overrides'), keep_env=keep_env)
    goals = config['goals']
    region = GoalRegion(x_from_edge=tuple(goals['x_from_edge']), y=tuple(goals['y']),
                        z_above_top=tuple(goals['z_above_top']))
    mdp = config['mdp']
    active = mdp.get('active')
    pool = GoalPool.build(scene, region, samples=int(goals['fk_samples']), voxel=float(goals['voxel']),
                          seed=0, cache_dir=cache_dir(), active=active, limits=mdp.get('limits'))
    n_joints = scene.n_joints if active is None else len(active)
    return Experiment(config, scene, make_bins(config), make_reward(config, n_joints), region, pool)
