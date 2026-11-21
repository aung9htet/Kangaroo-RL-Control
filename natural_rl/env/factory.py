"""Environment selection by action mode: one place decides which backend class a run, an evaluation or a video uses."""
from __future__ import annotations

from .reach_env import ReachBench


def make_env(scene, n: int, profile, spec, tasks: dict, horizon: int = 100, seed: int = 0, backend: str | None = None):
    """`ReachBench` (velocity and position commands) or `DynamicsBench` (torque), as the observation spec's action mode says.

    `backend` may force 'kinematic' or 'dynamics' (the spec's own backend is the default); 'physics' is built by the evaluator.
    """
    backend = backend or spec.backend
    if backend == 'dynamics':
        from .dynamics import DynamicsBench
        return DynamicsBench(scene, n, profile, spec, tasks, horizon=horizon, seed=seed)
    if backend == 'kinematic':
        if spec.action_mode == 'torque':
            raise ValueError('torque commands need the dynamics backend; the kinematic backend has no dynamics')
        return ReachBench(scene, n, profile, spec, tasks, horizon=horizon, seed=seed)
    raise ValueError(f'unknown backend {backend!r}')
