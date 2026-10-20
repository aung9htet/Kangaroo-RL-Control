"""Integration tests on the real LIBERO scene (run inside the container)."""
import mujoco
import numpy as np
import pytest

from kangaroo_rl.envs.libero import LiberoReachEnv
from kangaroo_rl.scene import PREFIX, ArmKinematics

pytestmark = pytest.mark.sim


def test_scene_facts(experiment):
    scene = experiment.scene
    assert scene.joint_names[:5] == tuple(f'arm_right_{i}_joint' for i in range(1, 6))
    assert scene.kinds == ('motor',) * 5 + ('position',) * 2
    assert np.isclose(scene.settings['base_clearance'], 0.25)
    assert np.isclose(scene.table_edge_base, 0.25)               # base sits 0.25 m behind the table edge
    palm = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + 'hand_right_palm_link')
    np.testing.assert_allclose(scene.model.body_pos[scene.reach_body], [-0.05, 0, 0])
    assert scene.model.body_parentid[scene.reach_body] == palm


def test_fk_parity_with_simulator_model(experiment, rng):
    scene = experiment.scene
    physics, data = scene.env.sim.model._model, mujoco.MjData(scene.env.sim.model._model)
    kinematics = ArmKinematics(scene)
    for _ in range(100):
        q = rng.uniform(scene.lower, scene.upper)
        data.qpos[:] = scene.rest_qpos
        data.qpos[scene.qpos_adr] = q
        mujoco.mj_kinematics(physics, data)
        np.testing.assert_allclose(kinematics.reach_point(q), scene.to_base(data.xpos[scene.reach_body]), atol=1e-9)


def test_collision_test_matches_full_contact_detection(experiment, rng):
    """Kinematics + collision on the filtered copy agrees with mj_forward on the simulator model,
    counting only right-arm contacts with the table or the robot."""
    scene = experiment.scene
    physics = scene.env.sim.model._model
    data = mujoco.MjData(physics)
    kinematics = ArmKinematics(scene)
    arm = set(scene.arm_geoms.tolist())
    table = mujoco.mj_name2id(physics, mujoco.mjtObj.mjOBJ_GEOM, 'table_collision')
    base = mujoco.mj_name2id(physics, mujoco.mjtObj.mjOBJ_BODY, PREFIX + 'base_link')

    def robot(geom):
        body = physics.geom_bodyid[geom]
        while body > 0:
            if body == base:
                return True
            body = physics.body_parentid[body]
        return False

    disagreements = 0
    for _ in range(200):
        q = rng.uniform(scene.lower, scene.upper)
        data.qpos[:] = scene.rest_qpos
        data.qpos[scene.qpos_adr] = q
        mujoco.mj_forward(physics, data)
        full = False
        for c in data.contact[:data.ncon]:
            g1, g2 = c.geom1, c.geom2
            if c.dist >= 0 or not (g1 in arm or g2 in arm):
                continue
            other = g2 if g1 in arm else g1
            if other == table or robot(other):
                full = True
                break
        disagreements += full != kinematics.evaluate(q)[1]
    assert disagreements == 0


def test_physics_adapter_follows_the_command(experiment):
    env = LiberoReachEnv(experiment.scene, experiment.bins, experiment.reward, 100, mode='A',
                         target_object='akita_black_bowl_2')
    q, goal = env.reset(experiment.eval_goals(1)[0])
    elbow, up = 3, list(experiment.bins.levels).index(0.3)
    action = np.full(7, experiment.bins.zero)
    action[elbow] = up                                           # +0.3 rad/s on the elbow
    for _ in range(20):
        step, _ = env.step(action)
    np.testing.assert_allclose(env.q_ref[elbow] - experiment.scene.home[elbow], 20 * 0.015, atol=1e-9)
    assert abs(env.measured()[elbow] - env.q_ref[elbow]) < 0.02    # the simulated joint tracks the reference


def test_object_appears_under_the_goal(experiment):
    env = LiberoReachEnv(experiment.scene, experiment.bins, experiment.reward, 100, mode='B',
                         target_object='akita_black_bowl_2', aim_offset=(0, 0, 0.15))
    goal = env.object_goal(np.array([0.40, -0.30]))
    env.reset(goal)
    np.testing.assert_allclose(env.object_position()[:2], goal[:2], atol=0.01)
    assert abs(env.object_position()[2] - (goal[2] - 0.15)) < 0.01


def test_training_smoke(reach_config, tmp_path):
    from kangaroo_rl.config import override
    from kangaroo_rl.train import train
    config = override(reach_config, ['train.steps=2000', 'train.warmup=500', 'train.eval_every=1000',
                                     'train.eval_goals=32', 'goals.fk_samples=200000', 'train.checkpoints=[1000]'])
    out = train(config, seed=0, out=tmp_path / 'run', device='cpu', threads=2)
    assert (out / 'metrics.csv').read_text().count('\n') >= 3
    assert (out / 'checkpoints' / 'step_00001000.pt').exists()
