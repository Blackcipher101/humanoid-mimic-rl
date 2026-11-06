"""Booster T1 base environment with full-body floor collisions (MJX)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.t1 import base as t1_base
from mujoco_playground._src.locomotion.t1 import t1_constants as t1_consts

from humanoid_mimic_rl import t1_model

# Contact sensors defined in playground's T1 sensor.xml.
_LEFT_FOOT_SENSORS = tuple(f"left_foot_{i}_floor_found" for i in range(1, 5))
_RIGHT_FOOT_SENSORS = tuple(f"right_foot_{i}_floor_found" for i in range(1, 5))


def noise_config() -> config_dict.ConfigDict:
    """Uniform observation noise half-widths (same values as playground's T1 joystick)."""
    return config_dict.create(
        level=1.0,  # 0 disables noise.
        scales=config_dict.create(
            joint_pos=0.03,  # rad
            joint_vel=1.5,  # rad/s
            gravity=0.05,  # unit vector components
            gyro=0.2,  # rad/s
        ),
    )


class T1FullCollisionEnv(t1_base.T1Env):
    """T1 on flat ground where trunk, head, arms and legs collide with the floor.

    Reuses playground's ``T1Env`` sensor accessors and model properties. Its
    ``__init__`` is intentionally not called because it loads the feet-only
    XML from disk; the model is built here from the derived full-collision
    XML instead (see ``t1_model.make_full_collision_xml``), with the same
    simulator settings as ``T1Env``.

    Required config keys: ``ctrl_dt``, ``sim_dt``, ``impl``, ``naconmax``,
    ``njmax``, ``soft_joint_pos_limit_factor``, ``noise_config``.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        mjx_env.MjxEnv.__init__(self, config, config_overrides)
        scene_xml, assets = t1_model.full_collision_flat_scene(t1_base.get_assets())
        self._model_assets = assets
        self._mj_model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.opt.ccd_iterations = 10
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = (t1_consts.ROOT_PATH / "xmls" / t1_model.FLAT_SCENE_XML_NAME).as_posix()
        self._init_common()

    # ------------------------------------------------------------------
    # Construction helpers.
    # ------------------------------------------------------------------

    def _init_common(self) -> None:
        m = self._mj_model
        joint_names = tuple(m.joint(i).name for i in range(1, m.njnt))
        if joint_names != t1_model.T1_JOINT_NAMES:
            raise RuntimeError(f"T1 joint order changed: {joint_names}")
        home = m.keyframe("home").qpos
        if not np.allclose(home[7:], t1_model.home_pose()):
            raise RuntimeError("T1 home keyframe differs from t1_model.T1_HOME_JOINT_POS")
        if m.nu != len(joint_names):
            raise RuntimeError(f"expected one actuator per joint, got nu={m.nu}")

        self._home_qpos = jp.array(home)
        self._default_pose = jp.array(home[7:])
        lowers, uppers = m.jnt_range[1:].T
        centre = 0.5 * (lowers + uppers)
        half = 0.5 * (uppers - lowers) * self._config.soft_joint_pos_limit_factor
        self._lowers = jp.array(lowers)
        self._uppers = jp.array(uppers)
        self._soft_lowers = jp.array(centre - half)
        self._soft_uppers = jp.array(centre + half)

        self._imu_site_id = m.site("imu").id
        floor = m.geom("floor").id
        colliding = [
            g
            for g in range(m.ngeom)
            if g != floor
            and (
                (m.geom_contype[g] & m.geom_conaffinity[floor])
                or (m.geom_conaffinity[g] & m.geom_contype[floor])
            )
        ]
        self._floor_geom_ids = np.array(colliding)
        self._floor_geom_rbound = jp.array(m.geom_rbound[colliding])
        fall_bodies = {m.body(b).id for b in t1_model.T1_FALL_BODIES}
        self._fall_geom_ids = np.array([g for g in colliding if m.geom_bodyid[g] in fall_bodies])
        if len(self._fall_geom_ids) != len(fall_bodies):
            raise RuntimeError("expected one floor-colliding geom per fall body")
        self._left_foot_sensor_adr = np.array([m.sensor_adr[m.sensor(s).id] for s in _LEFT_FOOT_SENSORS])
        self._right_foot_sensor_adr = np.array([m.sensor_adr[m.sensor(s).id] for s in _RIGHT_FOOT_SENSORS])

    # ------------------------------------------------------------------
    # State queries (all world frame unless stated otherwise).
    # ------------------------------------------------------------------

    def up_z(self, data: mjx.Data) -> jax.Array:
        """World z-component of the torso z-axis (1 upright, -1 upside-down)."""
        return self.get_gravity(data)[2]  # playground's "upvector" framezaxis sensor.

    def torso_height(self, data: mjx.Data) -> jax.Array:
        """Height (m) of the IMU site at the trunk origin."""
        return data.site_xpos[self._imu_site_id, 2]

    def feet_contact(self, data: mjx.Data) -> jax.Array:
        """``(2,)`` bool: any contact sphere of the left / right foot touches the floor."""
        left = jp.any(data.sensordata[self._left_foot_sensor_adr] > 0)
        right = jp.any(data.sensordata[self._right_foot_sensor_adr] > 0)
        return jp.stack([left, right])

    def fall_geom_min_height(self, data: mjx.Data) -> jax.Array:
        """Lowest centre height (m) of the trunk and head collision geoms."""
        return jp.min(data.geom_xpos[self._fall_geom_ids, 2])

    def lift_above_floor(self, data: mjx.Data, clearance_m: float) -> mjx.Data:
        """Raise the root so every floor-colliding geom's bounding sphere clears the floor.

        Prevents deep initial penetration (and the resulting explosive contact
        forces) when resetting to sampled or reference poses. Bounding spheres
        are conservative, so the robot may start up to a few cm in the air.
        """
        bottoms = data.geom_xpos[self._floor_geom_ids, 2] - self._floor_geom_rbound
        lift = jp.maximum(0.0, clearance_m - jp.min(bottoms))
        data = data.replace(qpos=data.qpos.at[2].add(lift))
        return mjx.forward(self.mjx_model, data)

    def is_invalid(self, data: mjx.Data) -> jax.Array:
        """True if the simulation state contains NaN/Inf."""
        return ~(jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel)))

    # ------------------------------------------------------------------
    # Simulation helpers.
    # ------------------------------------------------------------------

    def make_data(self, qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array) -> mjx.Data:
        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        return mjx.forward(self.mjx_model, data)

    def physics_step(self, data: mjx.Data, motor_targets: jax.Array) -> mjx.Data:
        """Advance one control step. Targets are clamped to joint ranges by MuJoCo's ctrlrange."""
        return mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)

    # ------------------------------------------------------------------
    # Observations.
    # ------------------------------------------------------------------

    @property
    def default_pose(self) -> jax.Array:
        """Home-pose joint angles (rad)."""
        return self._default_pose

    def obs_noise_half_widths(self) -> jax.Array:
        """Half-widths of the uniform noise ``proprio_obs`` adds to
        ``[gyro (3), gravity (3), joint pos (nu), joint vel (nu)]``."""
        s = self._config.noise_config.scales
        nu = self.action_size
        widths = jp.concatenate(
            [jp.full(3, s.gyro), jp.full(3, s.gravity), jp.full(nu, s.joint_pos), jp.full(nu, s.joint_vel)]
        )
        return widths * self._config.noise_config.level

    def _noisy(self, info: dict[str, Any], x: jax.Array, scale: float) -> jax.Array:
        info["rng"], key = jax.random.split(info["rng"])
        u = 2.0 * jax.random.uniform(key, shape=x.shape) - 1.0
        return x + u * self._config.noise_config.level * scale

    def proprio_obs(
        self, data: mjx.Data, info: dict[str, Any], extra: jax.Array
    ) -> dict[str, jax.Array]:
        """Actor (noisy, deployable sensors) and critic (privileged) observations.

        ``extra`` holds task features (e.g. phase) appended to both.
        """
        scales = self._config.noise_config.scales
        gyro = self.get_gyro(data)  # body frame, rad/s
        gravity_body = data.site_xmat[self._imu_site_id].T @ jp.array([0.0, 0.0, -1.0])
        joint_pos = data.qpos[7:]
        joint_vel = data.qvel[6:]
        state = jp.hstack(
            [
                self._noisy(info, gyro, scales.gyro),
                self._noisy(info, gravity_body, scales.gravity),
                self._noisy(info, joint_pos, scales.joint_pos) - self._default_pose,
                self._noisy(info, joint_vel, scales.joint_vel),
                info["last_act"],
                extra,
            ]
        )
        privileged = jp.hstack(
            [
                state,
                gyro,
                gravity_body,
                joint_pos - self._default_pose,
                joint_vel,
                self.get_local_linvel(data),
                self.get_global_angvel(data),
                self.get_accelerometer(data),
                self.torso_height(data),
                data.actuator_force,
                self.feet_contact(data).astype(jp.float32),
            ]
        )
        return {"state": state, "privileged_state": privileged}


def weighted_reward(
    terms: dict[str, jax.Array], scales: config_dict.ConfigDict, dt_s: float
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """``sum_k scale_k * term_k * dt``, clipped at 0 and with NaN/Inf mapped to 0.

    Returns the reward and the scaled terms (for metrics). Clipping at 0 follows
    playground's locomotion tasks so penalties cannot make early termination
    attractive.
    """
    scaled = {k: v * scales[k] for k, v in terms.items()}
    total = sum(scaled.values()) * dt_s
    total = jp.where(jp.isfinite(total), total, 0.0)
    return jp.clip(total, 0.0, 1e4), scaled
