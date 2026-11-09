"""Unitree G1 backflip tasks on the dynamically consistent reference clip.

* ``G1Backflip``: the hand-shaped keyframe/phase reward of ``backflip.BackflipTask``
  with its feed-forward prior (joint trajectory, root pitch, flight window)
  taken from the optimised clip (``scripts/make_g1_backflip.py``).
* ``G1Mimic``: DeepMimic tracking (``mimic.MimicTask``) of the same clip.

With ``config.torque_scale`` k != 1 the robot is a MODIFIED G1: all peak
torques are k x the published ones (``g1_model.torque_scale_label``); metrics
and videos carry that label.

Both run in torque mode (``G1FullCollisionEnv.TORQUE_MODE``): torque motors
limited to the G1 EDU peak torques (``g1_model.flip_peak_torques``) behind a
PD-to-torque layer, the same actuator limits the reference was optimised and
checked against.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from humanoid_mimic_rl import reference
from humanoid_mimic_rl.envs import backflip, mimic
from humanoid_mimic_rl.envs.g1_full_collision import G1FullCollisionEnv


def mimic_config(torque_scale: float) -> config_dict.ConfigDict:
    """DeepMimic config for the G1 clip with the clip's torque scale."""
    cfg = mimic.default_config()
    cfg.torque_scale = float(torque_scale)
    return cfg


def flight_window(clip: reference.ReferenceClip) -> tuple[float, float]:
    """Take-off and touch-down times (s) from the clip's foot-contact flags (one flight phase)."""
    if clip.foot_contact.size != clip.num_frames:
        raise ValueError("clip has no foot_contact flags; regenerate it with make_g1_backflip.py")
    air = np.flatnonzero(~clip.foot_contact)
    if air.size == 0 or air[-1] - air[0] + 1 != air.size:
        raise ValueError("clip must contain exactly one contiguous flight phase")
    return (air[0] - 1) / clip.fps_hz, (air[-1] + 1) / clip.fps_hz


def root_pitch_unwrapped(clip: reference.ReferenceClip) -> np.ndarray:
    """Root pitch (rad, about world y) per frame, unwrapped; assumes yaw = roll = 0."""
    q = clip.qpos[:, 3:7]
    return np.unwrap(2.0 * np.arctan2(q[:, 2], q[:, 0]))


def backflip_config(reference_path: str, torque_scale: float) -> config_dict.ConfigDict:
    """Backflip config for the G1 clip; ``torque_scale`` must be the clip's (see
    ``g1_model.resolve_torque_scale``)."""
    cfg = backflip.default_config()
    clip = reference.load_clip(Path(reference_path))
    cfg.reference_path = reference_path
    cfg.torque_scale = float(torque_scale)
    cfg.episode_length = int(math.ceil(clip.duration_s / cfg.ctrl_dt))
    # Torque term is normalised per joint by the peak torque (G1Backflip._torque_cost): dimensionless,
    # in [0, 29], the same at every torque scale. The default -1e-5 on raw N^2 m^2 (tuned for T1 servos)
    # gave ~-40 per step at 6x peak torques, so the clipped total reward was 0 everywhere and PPO had no
    # signal. -0.02: all 29 joints saturated costs 0.58 per step, against ~+1.5 for pose + orientation.
    cfg.reward_config.scales.torques = TORQUE_COST_SCALE_NORMALISED
    # Without a rotation budget the pitch_rate reward kept paying until the clip's touch-down, and the
    # PPO policy spun ~1.4 turns (9 rad) and never landed (0/32 rendered landings at 150M steps; 97-100%
    # falls in training evals after 250M more).
    cfg.rotation_target_rad = ROTATION_TARGET_RAD
    cfg.rotation_fade_rad = ROTATION_FADE_RAD
    cfg.over_rotation_limit_rad = OVER_ROTATION_LIMIT_RAD
    # Flight starts (training only, see G1Backflip.reset): probability of starting an episode in the clip's
    # ballistic flight instead of standing at t = 0, so touch-down and landing are practised directly.
    cfg.flight_start_prob = 0.0
    cfg.flight_start_after_takeoff_s = 0.05
    cfg.flight_start_before_touchdown_s = 0.15
    return cfg


# Reward scale of the peak-normalised torque term (see backflip_config).
TORQUE_COST_SCALE_NORMALISED = -0.02
# One somersault; the spin reward fades out over the last ROTATION_FADE_RAD before it.
ROTATION_TARGET_RAD = 2.0 * math.pi
ROTATION_FADE_RAD = 1.0
# Terminate (no further reward) once the robot is ~70 deg past a full turn.
OVER_ROTATION_LIMIT_RAD = 2.0 * math.pi + 1.2


class G1Backflip(backflip.BackflipTask, G1FullCollisionEnv):
    """G1 standing backflip, hand-shaped reward, prior from the optimised clip."""

    TORQUE_MODE = True

    def _torque_cost(self, data) -> jax.Array:
        """Sum over joints of (actuator torque / that joint's peak torque)^2 (dimensionless)."""
        return jp.sum(jp.square(data.actuator_force / self._peak_nm), axis=-1)

    def __init__(self, config: config_dict.ConfigDict | None = None, config_overrides: dict | None = None) -> None:
        super().__init__(config, config_overrides)
        c = self._config
        if not 0.0 <= c.flight_start_prob <= 1.0:
            raise ValueError(f"flight_start_prob must be in [0, 1], got {c.flight_start_prob}")
        clip = reference.load_clip(Path(c.reference_path))
        takeoff, landing = flight_window(clip)
        t_lo, t_hi = takeoff + c.flight_start_after_takeoff_s, landing - c.flight_start_before_touchdown_s
        if c.flight_start_prob > 0.0 and not t_lo < t_hi:
            raise ValueError(f"empty flight-start window [{t_lo:.3f}, {t_hi:.3f}] s (take-off {takeoff:.3f}, "
                             f"touch-down {landing:.3f})")
        self._flight_window_s = (t_lo, t_hi)
        self._clip_fps_hz = clip.fps_hz
        self._clip_qpos = jp.asarray(clip.qpos)
        self._clip_qvel = jp.asarray(clip.qvel)
        # Backward rotation accumulated by the clip since t = 0 (rad), to seed the rotation budget.
        pitch = root_pitch_unwrapped(clip)
        self._clip_back_rotation = jp.asarray(-(pitch - pitch[0]))

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Standing start at t = 0, or (with ``flight_start_prob``) the clip state at a random flight time.

        The clip's flight is ballistic (its inverse-dynamics check bounds the in-flight root wrench), so a
        flight start is a physically consistent airborne state; reward windows and phase features follow
        the clip time as in a standing start.
        """
        rng, k_flag, k_t = jax.random.split(rng, 3)
        state = super().reset(rng)
        c = self._config
        if c.flight_start_prob <= 0.0:
            return state
        t0 = jax.random.uniform(k_t, (), minval=self._flight_window_s[0], maxval=self._flight_window_s[1])
        k = jp.clip(jp.round(t0 * self._clip_fps_hz).astype(jp.int32), 0, self._clip_qpos.shape[0] - 1)
        t0 = k / self._clip_fps_hz
        qpos, qvel = self._clip_qpos[k], self._clip_qvel[k]
        data = self.make_data(qpos, qvel, qpos[7:])
        data = data.replace(time=jp.asarray(t0, data.time.dtype))
        info = dict(state.info)
        info["rotation_rad"] = self._clip_back_rotation[k]
        info["prev_root_quat"] = qpos[3:7]
        obs = self.proprio_obs(data, info, backflip.phase_features(data.time, self._motion_end_s))
        flight = jax.random.bernoulli(k_flag, c.flight_start_prob)
        pick = lambda a, b: jp.where(flight, a, b)  # noqa: E731
        return state.replace(
            data=jax.tree.map(pick, data, state.data),
            obs=jax.tree.map(pick, obs, state.obs),
            info=jax.tree.map(pick, info, state.info),
        )

    def _motion_prior(self) -> backflip.MotionPrior:
        clip = reference.load_clip(Path(self._config.reference_path))
        if clip.joint_names != self.JOINT_NAMES:
            raise ValueError("reference joint order does not match the G1 model")
        takeoff, landing = flight_window(clip)
        return backflip.MotionPrior(clip.fps_hz, clip.qpos[:, 7:], root_pitch_unwrapped(clip), takeoff, landing)


class G1Mimic(mimic.MimicTask, G1FullCollisionEnv):
    """G1 DeepMimic tracking of the optimised backflip clip."""

    TORQUE_MODE = True
