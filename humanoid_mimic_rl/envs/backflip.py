"""T1 standing backflip with a hand-shaped keyframe / phase reward.

No reference *clip* is tracked densely here (that is ``mimic.T1Mimic``).
Instead the keyframes in ``keyframes.t1_backflip_keyframes`` provide:

* a feed-forward joint trajectory ``q_kf(t)``: the policy outputs residual
  joint targets ``q_kf(t + dt) + action * action_scale``;
* phase windows derived from the keyframe contact flags (crouch/push-off,
  flight, landing) that switch reward terms on and off.

Reward terms (each in [0, 1] unless a cost):
  pose         exp(-g * |q - q_kf(t)|^2)       always (weak keyframe prior)
  orientation  exp(-g * angle(R, R_kf(t))^2)    root pitch follows keyframes
  takeoff      clip(v_z / v_target, 0, 1)       push-off window
  pitch_rate   clip(-omega_y / w_target, 0, 1)  push-off + flight window
  airborne     no foot contact                  flight window
  landing      upright, high, foot contact      after touch-down
  posture      gated home posture               settle window at the end
and the usual regularisers. The episode starts standing at t = 0 (no
reference state initialisation) and terminates early when the trunk or head
geom centre drops below ``fall_height_m`` (DeepMimic-style early
termination). The phase variable is ``data.time``, which Brax's cached
auto-reset restores together with the initial state.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from humanoid_mimic_rl import keyframes as kf
from humanoid_mimic_rl import rewards
from humanoid_mimic_rl import t1_model
from humanoid_mimic_rl.envs import t1_full_collision as base


def default_config() -> config_dict.ConfigDict:
    motion_end_s = t1_backflip_motion_end_s()
    ctrl_dt = 0.02
    return config_dict.create(
        ctrl_dt=ctrl_dt,  # s
        sim_dt=0.002,  # s
        episode_length=int(math.ceil(motion_end_s / ctrl_dt)),  # control steps
        action_repeat=1,
        action_scale=0.5,  # rad per unit action, residual on q_kf
        soft_joint_pos_limit_factor=0.95,
        keyframe_rate_hz=100.0,
        init_joint_noise_rad=0.05,
        # Phase windows relative to keyframe take-off / touch-down (s).
        pushoff_lead_s=0.25,
        airborne_margin_s=0.04,
        landing_delay_s=0.15,
        settle_delay_s=0.6,
        # Targets.
        takeoff_vz_target_mps=2.5,
        pitch_rate_target_radps=10.0,
        land_min_up_z=0.8,
        land_height_frac=0.75,  # of the home torso height
        fall_height_m=0.2,  # trunk/head geom centre height => fall
        pose_gain=0.3,
        orientation_gain=1.0,
        posture_gain=0.5,
        noise_config=base.noise_config(),
        reward_config=config_dict.create(
            scales=config_dict.create(
                pose=0.5,
                orientation=1.0,
                takeoff=1.0,
                pitch_rate=1.0,
                airborne=0.5,
                landing=2.0,
                posture=0.5,
                action_rate=-0.005,
                torques=-1e-5,
                dof_pos_limits=-1.0,
            ),
        ),
        impl="jax",
        naconmax=32 * 8192,
        njmax=160,
    )


def t1_backflip_motion_end_s() -> float:
    return kf.t1_backflip_keyframes()[-1].time_s


def phase_features(t_s: jax.Array, period_s: float) -> jax.Array:
    """``[phase, sin(2 pi phase), cos(2 pi phase)]`` with ``phase = clip(t / period, 0, 1)``."""
    phase = jp.clip(t_s / period_s, 0.0, 1.0)
    return jp.stack([phase, jp.sin(2.0 * jp.pi * phase), jp.cos(2.0 * jp.pi * phase)])


class T1Backflip(base.T1FullCollisionEnv):
    """Standing backflip on flat ground."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config or default_config(), config_overrides)
        keys = kf.t1_backflip_keyframes()
        kf.validate_keyframes(keys, t1_model.T1_JOINT_NAMES, t1_model.joint_ranges())
        traj = kf.interpolate_keyframes(
            keys, t1_model.T1_JOINT_NAMES, t1_model.home_pose(), self._config.keyframe_rate_hz
        )
        self._kf_rate_hz = float(self._config.keyframe_rate_hz)
        self._kf_joints = jp.array(traj.joint_pos_rad)
        self._kf_pitch = jp.array(traj.pitch_rad)
        self._motion_end_s = float(traj.times_s[-1])
        flight = kf.flight_phase(keys)
        c = self._config
        self._win_pushoff = (flight.takeoff_s - c.pushoff_lead_s, flight.takeoff_s + c.airborne_margin_s)
        self._win_rotate = (flight.takeoff_s - 0.1, flight.landing_s)
        self._win_airborne = (flight.takeoff_s + c.airborne_margin_s, flight.landing_s - c.airborne_margin_s)
        self._win_landing = (flight.landing_s + c.landing_delay_s, np.inf)
        self._win_settle = (flight.landing_s + c.settle_delay_s, np.inf)
        self._land_height_m = c.land_height_frac * t1_model.T1_HOME_ROOT_HEIGHT_M

    @property
    def motion_end_s(self) -> float:
        return self._motion_end_s

    def keyframe_at(self, t_s: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Interpolated keyframe joint angles (rad) and root pitch (rad) at ``t_s``."""
        n = self._kf_pitch.shape[0]
        f = jp.clip(t_s * self._kf_rate_hz, 0.0, n - 1.0)
        i0 = jp.clip(jp.floor(f).astype(jp.int32), 0, n - 2)
        a = f - i0
        joints = self._kf_joints[i0] * (1.0 - a) + self._kf_joints[i0 + 1] * a
        pitch = self._kf_pitch[i0] * (1.0 - a) + self._kf_pitch[i0 + 1] * a
        return joints, pitch

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_joint = jax.random.split(rng)
        noise = jax.random.uniform(
            k_joint, (self.action_size,), minval=-self._config.init_joint_noise_rad, maxval=self._config.init_joint_noise_rad
        )
        joints = jp.clip(self._default_pose + noise, self._lowers, self._uppers)
        qpos = self._home_qpos.at[7:].set(joints)
        data = self.make_data(qpos, jp.zeros(self.mjx_model.nv), joints)
        data = self.lift_above_floor(data, 0.0)
        data = data.replace(time=jp.zeros_like(data.time))
        info = {
            "rng": rng,
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales}
        metrics["fell"] = jp.zeros(())
        obs = self.proprio_obs(data, info, phase_features(data.time, self._motion_end_s))
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        c = self._config
        q_ff, _ = self.keyframe_at(state.data.time + self.dt)
        data = self.physics_step(state.data, q_ff + action * c.action_scale)

        t = data.time
        q_kf, pitch_kf = self.keyframe_at(t)
        up_z = self.up_z(data)
        height = self.torso_height(data)
        contact = self.feet_contact(data)
        angvel_body = data.qvel[3:6]  # free-joint angular velocity, body frame (rad/s)
        vz = data.qvel[2]  # world-frame vertical root velocity (m/s)

        def win(w: tuple[float, float]) -> jax.Array:
            return rewards.time_window(t, w[0], w[1], jp)

        landed = rewards.landing_reward(up_z, height, contact, c.land_min_up_z, self._land_height_m, jp)
        terms = {
            "pose": rewards.posture_reward(data.qpos[7:], q_kf, jp.ones(self.action_size), c.pose_gain, jp),
            "orientation": rewards.orientation_tracking_reward(
                data.qpos[3:7], rewards.pitch_quat(pitch_kf, jp), c.orientation_gain, jp
            ),
            "takeoff": win(self._win_pushoff) * rewards.ramp_reward(vz, c.takeoff_vz_target_mps, jp),
            "pitch_rate": win(self._win_rotate)
            * rewards.ramp_reward(rewards.backward_pitch_rate(angvel_body), c.pitch_rate_target_radps, jp),
            "airborne": win(self._win_airborne) * jp.where(jp.any(contact), 0.0, 1.0),
            "landing": win(self._win_landing) * landed,
            "posture": win(self._win_settle)
            * jp.where(landed > 0.0, 1.0, 0.0)
            * rewards.posture_reward(data.qpos[7:], self._default_pose, jp.ones(self.action_size), c.posture_gain, jp),
            "action_rate": rewards.action_rate_cost(action, state.info["last_act"], state.info["last_last_act"], jp),
            "torques": rewards.torque_cost(data.actuator_force, jp),
            "dof_pos_limits": rewards.joint_limit_cost(data.qpos[7:], self._soft_lowers, self._soft_uppers, jp),
        }
        reward, scaled = base.weighted_reward(terms, c.reward_config.scales, self.dt)

        fell = self.fall_geom_min_height(data) < c.fall_height_m
        done = fell | self.is_invalid(data)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        obs = self.proprio_obs(data, state.info, phase_features(t, self._motion_end_s))
        for k, v in scaled.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["fell"] = fell.astype(jp.float32)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))
