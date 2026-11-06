"""Unitree G1 base environment with full-body floor collisions (MJX).

Same interface as ``t1_full_collision.T1FullCollisionEnv`` (``up_z``,
``torso_height``, ``feet_contact``, ``fall_geom_min_height``,
``lift_above_floor``, ``proprio_obs``, ...), so the flip tasks are shared.
Observations are computed from the root state rather than sensors: the
gyro is the pelvis angular velocity in the pelvis frame (``qvel[3:6]``,
identical to playground's ``gyro_pelvis`` for a rigid body) and gravity is
expressed in the pelvis frame.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants

from humanoid_mimic_rl import g1_model


class G1FullCollisionEnv(mjx_env.MjxEnv):
    """G1 on flat ground; pelvis, torso, head, thighs, shins and hands collide with the floor."""

    HOME_ROOT_HEIGHT_M = g1_model.G1_HOME_ROOT_HEIGHT_M
    JOINT_NAMES = g1_model.G1_JOINT_NAMES
    END_EFFECTOR_SITES = g1_model.END_EFFECTOR_SITES

    def __init__(self, config: config_dict.ConfigDict, config_overrides: dict[str, Any] | None = None) -> None:
        super().__init__(config, config_overrides)
        scene_xml, assets = g1_model.full_collision_flat_scene(g1_base.get_assets())
        self._model_assets = assets
        self._mj_model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = (g1_constants.ROOT_PATH / "xmls" / g1_model.FLAT_SCENE_XML_NAME).as_posix()
        self._init_common()

    def _init_common(self) -> None:
        m = self._mj_model
        names = tuple(m.joint(i).name for i in range(1, m.njnt))
        if names != g1_model.G1_JOINT_NAMES:
            raise RuntimeError(f"G1 joint order changed: {names}")
        home = m.keyframe("home").qpos
        if not np.allclose(home[7:], g1_model.home_pose()):
            raise RuntimeError("G1 home keyframe differs from g1_model.G1_HOME_JOINT_POS")
        self._home_qpos = jp.array(home)
        self._default_pose = jp.array(home[7:])
        lowers, uppers = m.jnt_range[1:].T
        centre = 0.5 * (lowers + uppers)
        half = 0.5 * (uppers - lowers) * self._config.soft_joint_pos_limit_factor
        self._lowers, self._uppers = jp.array(lowers), jp.array(uppers)
        self._soft_lowers, self._soft_uppers = jp.array(centre - half), jp.array(centre + half)
        self._root_body_id = m.body(g1_model.ROOT_BODY).id
        floor_geoms = (*g1_model.FLOOR_PAIR_GEOMS, *g1_model.FOOT_GEOMS)
        self._floor_geom_ids = np.array([m.geom(g).id for g in floor_geoms])
        self._floor_geom_rbound = jp.array(m.geom_rbound[self._floor_geom_ids])
        self._fall_geom_ids = np.array([m.geom(g).id for g in g1_model.FALL_GEOMS])
        self._foot_sensor_adr = np.array([m.sensor_adr[m.sensor(s).id] for s in g1_model.FOOT_CONTACT_SENSORS])

    # State queries -----------------------------------------------------

    def _root_rot(self, data: mjx.Data) -> jax.Array:
        return data.xmat[self._root_body_id]

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        """Pelvis z-axis in the world frame (playground's "upvector" convention)."""
        return self._root_rot(data)[:, 2]

    def up_z(self, data: mjx.Data) -> jax.Array:
        return self._root_rot(data)[2, 2]

    def torso_height(self, data: mjx.Data) -> jax.Array:
        """Pelvis (root) height (m)."""
        return data.qpos[2]

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return self._root_rot(data).T @ data.qvel[0:3]

    def feet_contact(self, data: mjx.Data) -> jax.Array:
        return data.sensordata[self._foot_sensor_adr] > 0

    def fall_geom_min_height(self, data: mjx.Data) -> jax.Array:
        return jp.min(data.geom_xpos[self._fall_geom_ids, 2])

    def lift_above_floor(self, data: mjx.Data, clearance_m: float) -> mjx.Data:
        """Raise the root so all floor colliders' bounding spheres clear the floor (see T1 base)."""
        bottoms = data.geom_xpos[self._floor_geom_ids, 2] - self._floor_geom_rbound
        lift = jp.maximum(0.0, clearance_m - jp.min(bottoms))
        return mjx.forward(self.mjx_model, data.replace(qpos=data.qpos.at[2].add(lift)))

    def is_invalid(self, data: mjx.Data) -> jax.Array:
        return ~(jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel)))

    # Simulation ---------------------------------------------------------

    def make_data(self, qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array) -> mjx.Data:
        data = mjx_env.make_data(
            self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl, impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax, njmax=self._config.njmax,
        )
        return mjx.forward(self.mjx_model, data)

    def physics_step(self, data: mjx.Data, motor_targets: jax.Array) -> mjx.Data:
        return mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)

    # Observations -------------------------------------------------------

    @property
    def default_pose(self) -> jax.Array:
        return self._default_pose

    def obs_noise_half_widths(self) -> jax.Array:
        s = self._config.noise_config.scales
        nu = self.action_size
        widths = jp.concatenate(
            [jp.full(3, s.gyro), jp.full(3, s.gravity), jp.full(nu, s.joint_pos), jp.full(nu, s.joint_vel)]
        )
        return widths * self._config.noise_config.level

    def proprio_obs(self, data: mjx.Data, info: dict[str, Any], extra: jax.Array) -> dict[str, jax.Array]:
        """Actor obs ``[gyro, gravity_body, q - q0, qd, last_act, extra]`` (noisy) and critic obs."""
        rot = self._root_rot(data)
        gyro = data.qvel[3:6]
        gravity_body = rot.T @ jp.array([0.0, 0.0, -1.0])
        joint_pos, joint_vel = data.qpos[7:], data.qvel[6:]
        noise = self.obs_noise_half_widths()
        clean = jp.concatenate([gyro, gravity_body, joint_pos - self._default_pose, joint_vel])
        info["rng"], key = jax.random.split(info["rng"])
        noisy = clean + noise * (2.0 * jax.random.uniform(key, clean.shape) - 1.0)
        state = jp.hstack([noisy, info["last_act"], extra])
        privileged = jp.hstack(
            [
                state,
                clean,
                rot.T @ data.qvel[0:3],
                rot @ data.qvel[3:6],
                data.qpos[2],
                data.actuator_force,
                self.feet_contact(data).astype(jp.float32),
            ]
        )
        return {"state": state, "privileged_state": privileged}

    # Accessors ------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
