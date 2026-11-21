"""Raw simulator frames for the checkpoint videos: fixed camera, no overlays, markers, text or post-processing.

The renderer draws the ACCEPTED robot state (rejected and still frames included) from the compiled MuJoCo model of
the scene, with its own `MjData`; it never touches the optimizer, replay buffers or the environment that produced
the state. The goal is deliberately NOT drawn: goal identity is in the video manifest and the rollout parquet.
Needs MUJOCO_GL=egl (NVIDIA) or osmesa (software), which the Compose service sets.
"""
from __future__ import annotations

import hashlib

import mujoco
import numpy as np


class RawRenderer:
    def __init__(self, scene, render_cfg: dict):
        self.scene = scene
        self.cfg = render_cfg
        self.width, self.height = int(render_cfg['width']), int(render_cfg['height'])
        self.model = scene.model                           # compiled model copy: geometry and materials are unchanged
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.height)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        view = render_cfg['view']
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = view['lookat']
        self.camera.distance = float(view['distance'])
        self.camera.azimuth, self.camera.elevation = float(view['azimuth']), float(view['elevation'])
        self.options = mujoco.MjvOption()
        self.options.geomgroup[0] = int(bool(render_cfg['scene_option']['collision_geoms']))

    def render(self, q) -> np.ndarray:
        """RGB uint8 frame (height, width, 3) of the scene with the 7 controlled joints at q."""
        self.data.qpos[:] = self.scene.rest_qpos
        self.data.qpos[self.scene.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)             # positions for rendering only; no dynamics are run
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def frame_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame, dtype=np.uint8).tobytes()).hexdigest()
