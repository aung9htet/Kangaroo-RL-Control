"""Offscreen pictures of the Kangaroo_Libero scene with the goal, the hand's path and labels drawn in.

Needs MUJOCO_GL=egl (NVIDIA) or osmesa (software), which run_rl_docker.bash --headless sets.
"""
from __future__ import annotations

import mujoco
import numpy as np

GOAL_RGBA = (0.95, 0.25, 0.20, 0.9)
PATH_RGBA = (0.15, 0.55, 0.95, 0.9)
HAND_RGBA = (0.10, 0.80, 0.35, 0.9)


class SceneCamera:
    """Free camera in world coordinates (lookat, distance, azimuth, elevation in degrees)."""

    def __init__(self, lookat, distance, azimuth, elevation):
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = lookat
        self.camera.distance, self.camera.azimuth, self.camera.elevation = distance, azimuth, elevation


# Views used for the figures (world frame; the robot faces +x, the table is centred on the origin).
VIEWS = {
    'overview': dict(lookat=(-0.25, -0.05, 0.95), distance=2.4, azimuth=140, elevation=-22),
    'side': dict(lookat=(-0.35, -0.25, 1.0), distance=1.7, azimuth=90, elevation=-8),
    'reach': dict(lookat=(-0.30, -0.25, 1.0), distance=1.25, azimuth=155, elevation=-25),
    'top': dict(lookat=(-0.30, -0.15, 0.9), distance=1.9, azimuth=90, elevation=-89),
}


class Renderer:
    def __init__(self, model, data, width=960, height=720, view='overview', show_collision=False):
        self.model, self.data = model, data
        # LIBERO models default to a 640 x 480 offscreen buffer; enlarging it does not affect the physics.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.options = mujoco.MjvOption()
        self.options.geomgroup[0] = int(show_collision)
        self.set_view(view)

    def set_view(self, view):
        self.camera = SceneCamera(**(VIEWS[view] if isinstance(view, str) else view)).camera

    def _add(self, kind, size, position, rgba, rotation=None):
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.asarray(size, dtype=np.float64), np.asarray(position, dtype=np.float64),
                            np.eye(3).ravel() if rotation is None else rotation, np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1

    def _segment(self, start, end, width, rgba):
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(start, dtype=np.float64),
                             np.asarray(end, dtype=np.float64))
        scene.ngeom += 1

    def render(self, goals_world=(), path_world=None, hand_world=None, goal_radius=0.015, pointer=False):
        """Render the current state; goals are spheres, the path a polyline, the hand a small sphere."""
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        for goal in np.atleast_2d(goals_world) if len(goals_world) else []:
            self._add(mujoco.mjtGeom.mjGEOM_SPHERE, (goal_radius, 0, 0), goal, GOAL_RGBA)
        if path_world is not None and len(path_world) > 1:
            for a, b in zip(path_world[:-1], path_world[1:]):
                if np.linalg.norm(b - a) > 1e-6:
                    self._segment(a, b, 0.004, PATH_RGBA)
        if hand_world is not None:
            self._add(mujoco.mjtGeom.mjGEOM_SPHERE, (0.012, 0, 0), hand_world, HAND_RGBA)
            if pointer and len(goals_world):
                self._segment(hand_world, np.atleast_2d(goals_world)[0], 0.002, (1.0, 0.85, 0.1, 0.9))
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def joint_overlay(renderer: Renderer, scene, axis_length=0.10):
    """Render with the 7 controlled joint axes drawn as arrows at their anchors."""
    renderer.renderer.update_scene(renderer.data, camera=renderer.camera, scene_option=renderer.options)
    model, data = renderer.model, renderer.data
    colours = [(0.90, 0.30, 0.25, 1), (0.95, 0.60, 0.15, 1), (0.90, 0.85, 0.20, 1), (0.35, 0.75, 0.30, 1),
               (0.20, 0.65, 0.85, 1), (0.40, 0.40, 0.90, 1), (0.75, 0.35, 0.80, 1)]
    anchors = []
    for i, name in enumerate(scene.joint_names):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'robot0_' + name)
        anchor, axis = data.xanchor[joint].copy(), data.xaxis[joint].copy()
        renderer._segment(anchor - 0.5 * axis_length * axis, anchor + 0.5 * axis_length * axis, 0.006, colours[i])
        anchors.append(anchor)
    hand = data.xpos[scene.reach_body]
    renderer._add(mujoco.mjtGeom.mjGEOM_SPHERE, (0.014, 0, 0), hand, HAND_RGBA)
    return renderer.renderer.render().copy(), np.array(anchors), colours


def project(renderer: Renderer, points_world):
    """Pixel coordinates (u right, v down) of world points in the last rendered frame, for labels."""
    camera = renderer.renderer.scene.camera[0]
    forward, up = np.asarray(camera.forward), np.asarray(camera.up)
    right = np.cross(forward, up)
    height, width = renderer.renderer.height, renderer.renderer.width
    half_height = 0.5 * (camera.frustum_top - camera.frustum_bottom)
    middle = 0.5 * (camera.frustum_top + camera.frustum_bottom)
    half_width = half_height * width / height
    relative = np.atleast_2d(points_world) - np.asarray(camera.pos)
    depth = relative @ forward
    x = (relative @ right) * camera.frustum_near / depth
    y = (relative @ up) * camera.frustum_near / depth
    u = 0.5 * width * (1 + (x - camera.frustum_center) / half_width)
    v = 0.5 * height * (1 - (y - middle) / half_height)
    return np.stack([u, v], axis=1)
