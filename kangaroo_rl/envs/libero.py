"""The MDP executed by the simulator: Kangaroo_Libero's KangarooEnv (100 Hz PD control, MuJoCo physics).

One decision (dt = 0.05 s) is 5 simulator ticks. The commanded velocity is integrated into absolute
position targets, the only command KangarooEnv accepts:

    q_ref <- q_ref + (q_cmd - q_ref_start) / 5            (per tick, linear between decisions)
    target = clip(q_ref + tau_j * v)                       (velocity feed-forward)

For the arm motors KangarooEnv applies kp (target - q) - kd qdot + gravity compensation, which lags a
ramp by kd v / kp; tau = kd / kp cancels it (shoulder and elbow joints). The forearm roll is too light for
its explicitly applied kd (kd dt / M = 3.6), so it gets none; the wrist joints are MuJoCo position
actuators without a velocity term, so tau = 0 there too.

Two observation modes (the state is joint positions only, so the simulator is partially observed):
  A: the policy sees q_ref, the commanded positions, which evolve exactly by the model rule T;
  B: the policy sees the measured positions, and each decision integrates from them.
Accuracy is always measured on the simulated hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..mdp import Reward, TransitionRule, VelocityBins, step_model
from ..scene import PREFIX, ArmKinematics, Scene


@dataclass
class PhysicsStep:
    q_next: np.ndarray       # observed joint angles (q_ref in mode A, measured in mode B)
    collided: bool           # move rejected by the model rule
    point_next: np.ndarray   # measured reach point (base frame)
    reward: float
    distance: float
    tracking: float          # max |q_measured - q_ref| over the joints (rad)
    object_shift: float      # how far the target object has moved since it appeared (m)


class LiberoReachEnv:
    def __init__(self, scene: Scene, bins: VelocityBins, reward: Reward, horizon: int, mode='A', feedforward=True,
                 target_object=None, aim_offset=(0.0, 0.0, 0.15), park_others=True):
        if scene.env is None:
            raise ValueError('Build the scene with keep_env=True for the simulator')
        if mode not in ('A', 'B'):
            raise ValueError("mode must be 'A' or 'B'")
        self.scene, self.bins, self.reward, self.horizon, self.mode = scene, bins, reward, horizon, mode
        self.kenv = scene.env
        self.model, self.data = self.kenv.sim.model._model, self.kenv.sim.data._data
        self.ticks = int(round(bins.dt / self.kenv.dt))
        self.rule = TransitionRule(bins, scene.lower, scene.upper)
        self.kinematics = ArmKinematics(scene)
        self.home_targets = np.array(self.kenv.targets, copy=True)
        self.home_targets[scene.channels] = scene.home
        robot = self.kenv.robot
        self.tau = np.zeros(scene.n_joints)
        if feedforward:
            inertia = self._joint_inertia()
            for j, channel in enumerate(scene.channels):
                if scene.kinds[j] == 'motor':
                    # KangarooEnv applies kd explicitly every physics step; where kd * dt / M >= 1 (the forearm
                    # roll, M = 0.011 kg m^2) that damping is near its stability limit and a feed-forward lead
                    # excites a numerical oscillation, so those joints get no feed-forward.
                    if robot.kd[channel] * self.model.opt.timestep / inertia[j] < 1.0:
                        self.tau[j] = robot.kd[channel] / robot.kp[channel]
                else:
                    actuator = robot.command_actuators[channel]
                    self.tau[j] = -self.model.actuator_biasprm[actuator, 2] / self.model.actuator_gainprm[actuator, 0]
        self.aim_offset = np.asarray(aim_offset, dtype=np.float64)
        self.objects = self._free_objects()
        self.target_object = target_object
        self.park_others = park_others
        self.q_ref = scene.home.copy()
        self.goal = np.zeros(3)
        self.object_start = None
        self.t = 0

    def _joint_inertia(self):
        """Diagonal of the joint-space inertia matrix at the home pose, for the 7 joints."""
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.scene.rest_qpos
        mujoco.mj_forward(self.model, data)
        return np.array([data.qM[self.model.dof_Madr[a]] for a in self.scene.dof_adr])

    # ----------------------------------------------------------------- scene objects
    def _free_objects(self):
        objects = {}
        for joint in range(self.model.njnt):
            if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                body = self.model.jnt_bodyid[joint]
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
                objects[name.removesuffix('_main')] = (body, self.model.jnt_qposadr[joint], self.model.jnt_dofadr[joint])
        return objects

    def object_rest_height(self, name=None) -> float:
        """Base-frame height of the object's origin when it rests on the table (from the settled scene)."""
        _, adr, _ = self.objects[name or self.target_object]
        return float(self.scene.to_base(self.scene.rest_qpos[adr:adr + 3])[2])

    def object_goal(self, xy_base, name=None):
        """The goal for an object appearing at (x, y): its rest origin plus the aim offset."""
        z = self.object_rest_height(name)
        return np.array([xy_base[0], xy_base[1], z]) + self.aim_offset

    def _place(self, name, position_world):
        _, adr, dof = self.objects[name]
        self.data.qpos[adr:adr + 3] = position_world
        self.data.qvel[dof:dof + 6] = 0.0

    def object_position(self, name=None):
        body = self.objects[name or self.target_object][0]
        return self.scene.to_base(self.data.xpos[body])

    # ----------------------------------------------------------------- episode
    def reset(self, goal, q0=None, settle_ticks=10):
        """Robot at q0 (default home), everything still; the target object appears under the goal."""
        scene = self.scene
        q0 = scene.home if q0 is None else np.asarray(q0, dtype=np.float64)
        self.data.qpos[:] = scene.rest_qpos
        self.data.qvel[:] = 0.0
        self.data.qpos[scene.qpos_adr] = q0
        self.goal = np.asarray(goal, dtype=np.float64).copy()
        if self.target_object is not None:
            self._park(exclude=self.target_object)
            self._place(self.target_object, scene.to_world(self.goal - self.aim_offset))
        mujoco.mj_forward(self.model, self.data)
        self.q_ref = q0.copy()
        targets = self.home_targets.copy()
        targets[scene.channels] = q0
        for _ in range(settle_ticks):
            self.kenv.step(targets)
        self.object_start = self.object_position() if self.target_object is not None else None
        self.t = 0
        return self.observe()

    def _park(self, exclude):
        """Move the other free objects to the far side of the table, out of the arm's reach."""
        if not self.park_others:
            return
        others = [name for name in self.objects if name != exclude]
        x0, x1, y0, y1, top = self.scene.table_box_base()
        for i, name in enumerate(others):
            _, adr, _ = self.objects[name]
            y = y0 + (i + 1) * (y1 - y0) / (len(others) + 1)
            rest = self.scene.to_base(self.scene.rest_qpos[adr:adr + 3])
            self._place(name, self.scene.to_world(np.array([x1 - 0.15, y, rest[2]])))

    def retarget(self, goal):
        """The object reappears elsewhere; the arm continues from where it is."""
        self.goal = np.asarray(goal, dtype=np.float64).copy()
        if self.target_object is not None:
            self._place(self.target_object, self.scene.to_world(self.goal - self.aim_offset))
            mujoco.mj_forward(self.model, self.data)
            self.object_start = self.object_position()
        self.t = 0
        return self.observe()

    def measured(self):
        return self.data.qpos[self.scene.qpos_adr].copy()

    def observe(self):
        q = self.q_ref.copy() if self.mode == 'A' else self.measured()
        return q, self.goal.copy()

    def reach_point(self):
        return self.scene.to_base(self.data.xpos[self.scene.reach_body])

    def step(self, actions, on_tick=None) -> tuple[PhysicsStep, bool]:
        actions = np.asarray(actions).reshape(-1)
        before = self.reach_point()
        start = self.q_ref if self.mode == 'A' else self.measured()
        command, collided, _ = step_model(self.rule, self.kinematics, start, actions)
        velocity = (command - start) / self.bins.dt          # what the rule actually commands
        targets = self.home_targets.copy()
        for tick in range(1, self.ticks + 1):
            reference = start + (command - start) * tick / self.ticks
            targets[self.scene.channels] = np.clip(reference + self.tau * velocity, self.scene.lower, self.scene.upper)
            self.kenv.step(targets)
            if on_tick is not None:
                on_tick(self)
        self.q_ref = command
        point = self.reach_point()
        distance = float(np.linalg.norm(point - self.goal))
        reward = float(self.reward(point, self.goal, collided, self.rule.velocity(actions), before))
        shift = 0.0 if self.object_start is None else float(np.linalg.norm(self.object_position() - self.object_start))
        self.t += 1
        q_obs = self.observe()[0]
        tracking = float(np.max(np.abs(self.measured() - command)))
        return PhysicsStep(q_obs, collided, point, reward, distance, tracking, shift), self.t >= self.horizon
