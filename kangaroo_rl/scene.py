"""The Kangaroo arm as the RL problem sees it: 7 joints, limits, reach point, table and collisions.

One LIBERO scene is built through Kangaroo_Libero's `make_env`. Everything the learner needs is read
from that scene, so the model used for training and the simulator used for evaluation are the same
compiled MuJoCo model.
"""
from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

PREFIX = 'robot0_'
ARM_JOINTS = ('arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint',
              'arm_right_5_joint', 'hand_right_palm_roll_joint', 'hand_right_palm_pitch_joint')
ARM_ROOT_BODY = 'arm_right_1_link'
REACH_BODY = 'hand_right_manipulation_link'   # PAL's manipulation frame, 5 cm along the fingers
BASE_BODY = 'base_link'
TABLE_GEOM = 'table_collision'
# Collision bits in the kinematic copy: right-arm geoms collide with each other and with obstacles
# (table, rest of the robot); obstacles never collide with each other; everything else is off.
ARM_BITS = (2, 3)
OBSTACLE_BITS = (1, 2)

_OBJ = mujoco.mjtObj


def write_sim_settings(overrides: dict | None) -> Path:
    """Write the simulator settings with `overrides` applied and point KANGAROO_CONFIG at them.

    The source is the submodule's `config/simulation.yaml`; Kangaroo_Libero itself is not changed.
    """
    repo = Path(os.environ.get('KANGAROO_REPO', '/workspace/Kangaroo_Libero'))
    source = repo / 'config/simulation.yaml'
    if not source.exists():
        source = Path('/opt/kangaroo-libero/config/simulation.yaml')
    settings = yaml.safe_load(source.read_text())
    settings.update(overrides or {})
    text = yaml.safe_dump(settings, sort_keys=True)
    digest = hashlib.sha1(text.encode()).hexdigest()[:10]
    runtime = Path(os.environ.get('KANGAROO_RL_REPO', Path(__file__).resolve().parents[1])) / '.runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    path = runtime / f'simulation-{digest}.yaml'
    if not path.exists():
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(text)
        temporary.replace(path)
    os.environ['KANGAROO_CONFIG'] = str(path)
    return path


def _name2id(model, kind, name):
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise KeyError(f'{name} not found in the model')
    return index


def _subtree(model, root: int) -> set[int]:
    bodies = set()
    for body in range(model.nbody):
        walk = body
        while walk > 0:
            if walk == root:
                bodies.add(body)
                break
            walk = model.body_parentid[walk]
    return bodies


@dataclass
class Scene:
    """Facts about one LIBERO scene with Kangaroo, expressed for the 7 controlled joints."""
    model: mujoco.MjModel          # kinematic copy with filtered collision bits
    rest_qpos: np.ndarray          # full qpos of the settled home pose (arm exactly at home)
    joint_names: tuple
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    channels: np.ndarray           # indices of the 7 joints in KangarooEnv.joint_names (38)
    lower: np.ndarray
    upper: np.ndarray
    home: np.ndarray
    kinds: tuple                   # 'motor' (software PD) or 'position' (MuJoCo actuator) per joint
    reach_body: int
    base_pos: np.ndarray
    base_rot: np.ndarray
    table_center: np.ndarray       # world frame
    table_half: np.ndarray
    arm_geoms: np.ndarray
    ignored: np.ndarray            # ngeom x ngeom: pairs already in contact at home
    settings: dict
    settings_path: str
    suite: str
    task_id: int
    seed: int
    env: object = field(default=None, repr=False)   # KangarooEnv when kept for physics

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def to_base(self, points_world):
        return (np.asarray(points_world) - self.base_pos) @ self.base_rot

    def to_world(self, points_base):
        return np.asarray(points_base) @ self.base_rot.T + self.base_pos

    @property
    def table_top_base(self) -> float:
        """Height of the table top in the base frame."""
        top = self.table_center + np.array([0.0, 0.0, self.table_half[2]])
        return float(self.to_base(top)[2])

    @property
    def table_edge_base(self) -> float:
        """x of the table's near edge (the side facing the robot) in the base frame."""
        edge = self.table_center - np.array([self.table_half[0], 0.0, 0.0])
        return float(self.to_base(edge)[0])

    def table_box_base(self):
        """Table top rectangle in the base frame: (x0, x1, y0, y1, z_top)."""
        corners = self.to_base(self.table_center + np.array([[-1, -1, 1], [1, 1, 1]]) * self.table_half)
        return (*np.sort(corners[:, 0]), *np.sort(corners[:, 1]), self.table_top_base)

    def signature(self) -> str:
        """Short hash of everything that defines the kinematic problem (used for caches)."""
        parts = [self.suite, str(self.task_id), str(self.lower.round(6)), str(self.upper.round(6)),
                 str(self.home.round(6)), str(self.base_pos.round(6)), str(self.table_center.round(6)),
                 str(self.table_half.round(6)), yaml.safe_dump(self.settings, sort_keys=True)]
        return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:12]


def build_scene(suite='libero_spatial', task_id=0, seed=0, sim_overrides=None, keep_env=False,
                settle_steps=100, joints=ARM_JOINTS) -> Scene:
    """Create the LIBERO scene, settle the home pose and extract the arm problem."""
    settings_path = write_sim_settings(sim_overrides)
    from kangaroo_libero import make_env   # imported after KANGAROO_CONFIG is set

    env = make_env(suite=suite, task_id=task_id, seed=seed)
    env.reset()
    home_targets = env.targets
    for _ in range(settle_steps):
        env.step(home_targets)
    physics = env.sim.model._model
    data = env.sim.data._data

    channels = np.array([env.joint_names.index(name) for name in joints])
    joint_ids = [_name2id(physics, _OBJ.mjOBJ_JOINT, PREFIX + name) for name in joints]
    qpos_adr = np.array([physics.jnt_qposadr[j] for j in joint_ids])
    dof_adr = np.array([physics.jnt_dofadr[j] for j in joint_ids])
    lower, upper = env.joint_limits[channels, 0].copy(), env.joint_limits[channels, 1].copy()
    home = home_targets[channels].copy()
    kinds = tuple(env.robot.channels[i]['kind'] for i in channels)
    rest_qpos = data.qpos.copy()
    rest_qpos[qpos_adr] = home

    base = _name2id(physics, _OBJ.mjOBJ_BODY, PREFIX + BASE_BODY)
    table = _name2id(physics, _OBJ.mjOBJ_GEOM, TABLE_GEOM)
    if not np.allclose(data.geom_xmat[table].reshape(3, 3), np.eye(3), atol=1e-9):
        raise ValueError('Expected an axis-aligned table top')

    model = copy.deepcopy(physics)
    arm_bodies = _subtree(model, _name2id(model, _OBJ.mjOBJ_BODY, PREFIX + ARM_ROOT_BODY))
    robot_bodies = _subtree(model, base)
    collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    in_arm = np.array([model.geom_bodyid[g] in arm_bodies for g in range(model.ngeom)])
    in_robot = np.array([model.geom_bodyid[g] in robot_bodies for g in range(model.ngeom)])
    arm_geoms = np.flatnonzero(collidable & in_arm)
    obstacles = np.flatnonzero(collidable & in_robot & ~in_arm)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.geom_contype[arm_geoms], model.geom_conaffinity[arm_geoms] = ARM_BITS
    model.geom_contype[obstacles], model.geom_conaffinity[obstacles] = OBSTACLE_BITS
    model.geom_contype[table], model.geom_conaffinity[table] = OBSTACLE_BITS

    probe = mujoco.MjData(model)
    probe.qpos[:] = rest_qpos
    mujoco.mj_kinematics(model, probe)
    mujoco.mj_collision(model, probe)
    ignored = np.zeros((model.ngeom, model.ngeom), dtype=bool)
    count = probe.ncon
    g1, g2 = probe.contact.geom1[:count], probe.contact.geom2[:count]
    touching = probe.contact.dist[:count] < 0
    ignored[g1[touching], g2[touching]] = True
    ignored[g2[touching], g1[touching]] = True

    scene = Scene(
        model=model, rest_qpos=rest_qpos, joint_names=tuple(joints), qpos_adr=qpos_adr, dof_adr=dof_adr,
        channels=channels, lower=lower, upper=upper, home=home, kinds=kinds,
        reach_body=_name2id(model, _OBJ.mjOBJ_BODY, PREFIX + REACH_BODY),
        base_pos=data.xpos[base].copy(), base_rot=data.xmat[base].reshape(3, 3).copy(),
        table_center=data.geom_xpos[table].copy(), table_half=physics.geom_size[table].copy(),
        arm_geoms=arm_geoms, ignored=ignored, settings=dict(env.settings), settings_path=str(settings_path),
        suite=suite, task_id=task_id, seed=seed)
    if keep_env:
        scene.env = env
    else:
        env.close()
    return scene


class ArmKinematics:
    """Reach-point forward kinematics and the collision test on one MjData of the kinematic model."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.model = scene.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = scene.rest_qpos
        self._jacp = np.zeros((3, self.model.nv))

    def evaluate(self, q, check_collision=True):
        """Reach point (base frame) at joint angles q, and whether q puts the arm into collision."""
        data = self.data
        data.qpos[self.scene.qpos_adr] = q
        mujoco.mj_kinematics(self.model, data)
        point = self.scene.to_base(data.xpos[self.scene.reach_body])
        if not check_collision:
            return point, False
        mujoco.mj_collision(self.model, data)
        count = data.ncon
        if count == 0:
            return point, False
        g1, g2 = data.contact.geom1[:count], data.contact.geom2[:count]
        hits = (data.contact.dist[:count] < 0) & ~self.scene.ignored[g1, g2]
        return point, bool(hits.any())

    def reach_point(self, q):
        return self.evaluate(q, check_collision=False)[0]

    def jacobian(self, q):
        """d(reach point)/dq in the base frame, 3 x 7 (after the last `evaluate` at the same q)."""
        data = self.data
        data.qpos[self.scene.qpos_adr] = q
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)
        mujoco.mj_jac(self.model, data, self._jacp, None, data.xpos[self.scene.reach_body], self.scene.reach_body)
        return self.scene.base_rot.T @ self._jacp[:, self.scene.dof_adr]
