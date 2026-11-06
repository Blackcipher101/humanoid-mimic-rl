"""Unitree G1 backflip tasks on the dynamically consistent reference clip.

* ``G1Backflip``: the hand-shaped keyframe/phase reward of ``backflip.BackflipTask``
  with its feed-forward prior (joint trajectory, root pitch, flight window)
  taken from the optimised clip (``scripts/make_g1_backflip.py``).
* ``G1Mimic``: DeepMimic tracking (``mimic.MimicTask``) of the same clip.

Both run in torque mode (``G1FullCollisionEnv.TORQUE_MODE``): torque motors
limited to the G1 EDU peak torques (``g1_model.flip_peak_torques``) behind a
PD-to-torque layer, the same actuator limits the reference was optimised and
checked against.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from ml_collections import config_dict

from humanoid_mimic_rl import reference
from humanoid_mimic_rl.envs import backflip, mimic
from humanoid_mimic_rl.envs.g1_full_collision import G1FullCollisionEnv


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


def backflip_config(reference_path: str) -> config_dict.ConfigDict:
    cfg = backflip.default_config()
    clip = reference.load_clip(Path(reference_path))
    cfg.reference_path = reference_path
    cfg.episode_length = int(math.ceil(clip.duration_s / cfg.ctrl_dt))
    return cfg


class G1Backflip(backflip.BackflipTask, G1FullCollisionEnv):
    """G1 standing backflip, hand-shaped reward, prior from the optimised clip."""

    TORQUE_MODE = True

    def _motion_prior(self) -> backflip.MotionPrior:
        clip = reference.load_clip(Path(self._config.reference_path))
        if clip.joint_names != self.JOINT_NAMES:
            raise ValueError("reference joint order does not match the G1 model")
        takeoff, landing = flight_window(clip)
        return backflip.MotionPrior(clip.fps_hz, clip.qpos[:, 7:], root_pitch_unwrapped(clip), takeoff, landing)


class G1Mimic(mimic.MimicTask, G1FullCollisionEnv):
    """G1 DeepMimic tracking of the optimised backflip clip."""

    TORQUE_MODE = True
