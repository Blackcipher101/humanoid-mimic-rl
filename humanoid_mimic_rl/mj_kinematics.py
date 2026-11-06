"""MuJoCo forward kinematics for ``clip_builder`` (requires the ``mujoco`` package)."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from humanoid_mimic_rl.clip_builder import FKResult


class MjKinematics:
    """Forward kinematics of a floating-base MuJoCo model.

    Args:
      model: compiled model; its first joint must be the free joint.
      root_body: body whose subtree CoM is the whole-robot CoM.
      foot_sphere_geoms: sphere geoms whose lowest points define foot contact.
      site_names: end-effector sites reported in ``FKResult.site_pos``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        root_body: str,
        foot_sphere_geoms: Sequence[str],
        site_names: Sequence[str],
    ) -> None:
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("first joint must be a free joint")
        self._model = model
        self._data = mujoco.MjData(model)
        self._root_body_id = model.body(root_body).id
        self._foot_geom_ids = np.array([model.geom(g).id for g in foot_sphere_geoms])
        for gid in self._foot_geom_ids:
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SPHERE:
                raise ValueError(f"foot geom {model.geom(gid).name} is not a sphere")
        self._foot_radius_m = model.geom_size[self._foot_geom_ids, 0]
        self._site_ids = np.array([model.site(s).id for s in site_names])
        self.site_names = tuple(site_names)

    def forward(self, qpos: np.ndarray) -> FKResult:
        if qpos.shape != (self._model.nq,):
            raise ValueError(f"qpos shape {qpos.shape} != ({self._model.nq},)")
        self._data.qpos[:] = qpos
        self._data.qvel[:] = 0.0
        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)
        centres = self._data.geom_xpos[self._foot_geom_ids].copy()
        centres[:, 2] -= self._foot_radius_m
        return FKResult(
            com=self._data.subtree_com[self._root_body_id].copy(),
            foot_points=centres,
            site_pos=self._data.site_xpos[self._site_ids].copy(),
        )
