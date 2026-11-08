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

    def _motion_prior(self) -> backflip.MotionPrior:
        clip = reference.load_clip(Path(self._config.reference_path))
        if clip.joint_names != self.JOINT_NAMES:
            raise ValueError("reference joint order does not match the G1 model")
        takeoff, landing = flight_window(clip)
        return backflip.MotionPrior(clip.fps_hz, clip.qpos[:, 7:], root_pitch_unwrapped(clip), takeoff, landing)


class G1Mimic(mimic.MimicTask, G1FullCollisionEnv):
    """G1 DeepMimic tracking of the optimised backflip clip."""

    TORQUE_MODE = True
