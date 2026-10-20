"""Reference controllers with the same interface as the agent: policy(q, g) -> actions.

- RandomPolicy: uniform random bins (the floor).
- DLSPolicy: damped least squares on the reach-point Jacobian, dq = J^T (J J^T + lambda^2 I)^-1 (w - p) / T,
  towards a scripted waypoint w: from below the table top it first lifts the hand, then approaches over
  the edge at a safe height, then descends onto the goal (a straight line from home runs into the table
  edge). It uses the model's kinematics and the table geometry, which the RL agent never sees.
  Quantised to the velocity bins it shows what the action discretisation allows; continuous (float
  velocities) it is the precision ceiling and the feasibility check for goals.
"""
from __future__ import annotations

import numpy as np

from .mdp import VelocityBins
from .scene import ArmKinematics, Scene


class RandomPolicy:
    def __init__(self, n_joints, bins: VelocityBins, seed=0):
        self.n_joints, self.bins = n_joints, bins
        self.rng = np.random.default_rng(seed)

    def __call__(self, q, goal):
        return self.rng.integers(self.bins.n_bins, size=(len(q), self.n_joints))


class DLSPolicy:
    def __init__(self, scene: Scene, bins: VelocityBins, quantise=True, time_constant=0.25, damping=0.05,
                 waypoint=True, lift=0.06, safe=0.08):
        self.kinematics = ArmKinematics(scene)
        self.bins, self.quantise, self.waypoint = bins, quantise, waypoint
        self.time_constant, self.damping = time_constant, damping
        self.edge, self.top = scene.table_edge_base, scene.table_top_base
        self.lift, self.safe = lift, safe

    def target(self, point, goal):
        """Lift first, cross the edge at a safe height, then go to the goal."""
        if not self.waypoint or goal[0] <= self.edge or point[0] >= self.edge + 0.03:
            return goal
        if point[2] < self.top + self.lift:
            return np.array([min(point[0], self.edge - 0.06), goal[1], self.top + 2 * self.lift])
        return np.array([goal[0], goal[1], max(goal[2], self.top + self.safe)])

    def velocity(self, q, goal):
        point = self.kinematics.reach_point(q)
        jacobian = self.kinematics.jacobian(q)
        desired = (self.target(point, np.asarray(goal)) - point) / self.time_constant
        gram = jacobian @ jacobian.T + self.damping ** 2 * np.eye(3)
        dq = jacobian.T @ np.linalg.solve(gram, desired)
        return dq / max(1.0, np.max(np.abs(dq)) / self.bins.v_max)

    def __call__(self, q, goal):
        velocities = np.array([self.velocity(qi, gi) for qi, gi in zip(q, goal)])
        if not self.quantise:
            return velocities
        # Error feedback: the rounding residual of each joint is carried into its next command, so the
        # binned velocities track the continuous ones on average instead of losing small components.
        if getattr(self, '_residual', None) is None or self._residual.shape != velocities.shape:
            self._residual = np.zeros_like(velocities)
        wanted = velocities + self._residual
        actions = self.bins.nearest(wanted)
        self._residual = wanted - self.bins.velocity(actions)
        return actions
