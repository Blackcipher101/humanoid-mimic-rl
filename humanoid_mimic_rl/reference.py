"""Reference motion clips: storage, validation, velocities and sampling.

A clip is a time series of generalized positions of a floating-base robot
sampled at a fixed rate:

* ``qpos``: ``(T, 7 + n_joints)`` = root position (m, world), root quaternion
  ``(w, x, y, z)`` (world <- body), joint angles (rad) in model order;
* ``qvel``: ``(T, 6 + n_joints)`` following MuJoCo's free-joint convention:
  root linear velocity in the **world** frame (m/s), root angular velocity in
  the **body** frame (rad/s), then joint velocities (rad/s);
* ``site_pos``: optional ``(T, n_sites, 3)`` world positions (m) of tracked
  end-effector sites, produced by forward kinematics;
* ``foot_contact``: optional ``(T,)`` bool, feet on the ground (used by
  dynamically generated clips; empty when unknown);
* ``licence`` / ``source``: provenance strings. Loading refuses clips without
  a licence string so unlicensed data cannot slip into training silently.

Offline processing is numpy-only. ``sample_clip`` is backend-agnostic so the
environment can call it inside ``jax.jit``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from humanoid_mimic_rl import quat

Array = Any

ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6
_QUAT_NORM_TOL = 1e-3


class ClipError(ValueError):
    """Raised when a reference clip is malformed."""


@dataclasses.dataclass(frozen=True)
class ReferenceClip:
    """A validated reference motion clip (see module docstring for layout)."""

    fps_hz: float
    qpos: np.ndarray
    qvel: np.ndarray
    joint_names: tuple[str, ...]
    site_names: tuple[str, ...]
    site_pos: np.ndarray
    licence: str
    source: str
    foot_contact: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration_s(self) -> float:
        """Time of the last frame (s); the first frame is at t = 0."""
        return (self.num_frames - 1) / self.fps_hz

    def validate(self) -> None:
        """Raise ``ClipError`` if shapes, values or provenance are invalid."""
        if not np.isfinite(self.fps_hz) or self.fps_hz <= 0.0:
            raise ClipError(f"fps_hz must be positive, got {self.fps_hz}")
        n_j = len(self.joint_names)
        if self.qpos.ndim != 2 or self.qpos.shape[1] != ROOT_QPOS_DIM + n_j:
            raise ClipError(
                f"qpos shape {self.qpos.shape} does not match {n_j} joints + 7 root dims"
            )
        if self.num_frames < 2:
            raise ClipError("a clip needs at least 2 frames")
        if self.qvel.shape != (self.num_frames, ROOT_QVEL_DIM + n_j):
            raise ClipError(
                f"qvel shape {self.qvel.shape} != ({self.num_frames}, {ROOT_QVEL_DIM + n_j})"
            )
        if self.site_pos.shape != (self.num_frames, len(self.site_names), 3):
            raise ClipError(
                f"site_pos shape {self.site_pos.shape} != "
                f"({self.num_frames}, {len(self.site_names)}, 3)"
            )
        for name, arr in (("qpos", self.qpos), ("qvel", self.qvel), ("site_pos", self.site_pos)):
            if not np.all(np.isfinite(arr)):
                raise ClipError(f"{name} contains NaN or Inf")
        qn = np.linalg.norm(self.qpos[:, 3:7], axis=-1)
        if np.max(np.abs(qn - 1.0)) > _QUAT_NORM_TOL:
            raise ClipError(
                f"root quaternion norm deviates from 1 by {np.max(np.abs(qn - 1.0)):.2e}"
            )
        if self.foot_contact.size not in (0, self.num_frames) or self.foot_contact.ndim != 1:
            raise ClipError(f"foot_contact shape {self.foot_contact.shape} != ({self.num_frames},)")
        if not self.licence.strip():
            raise ClipError("clip has no licence string; refusing to use it")


def finite_difference_qvel(qpos: np.ndarray, fps_hz: float) -> np.ndarray:
    """Velocities consistent with MuJoCo free-joint conventions.

    Central differences in the interior, one-sided at the ends. Angular
    velocity is expressed in the body frame: ``omega_k = log(q_k^-1 q_k+1) * fps``
    (forward) averaged with the backward estimate, which is exact for constant
    body-rate rotations.
    """
    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ClipError("qpos must be (T >= 2, nq)")
    dt = 1.0 / fps_hz
    n = qpos.shape[0]

    lin = np.gradient(qpos[:, 0:3], dt, axis=0)
    joints = np.gradient(qpos[:, ROOT_QPOS_DIM:], dt, axis=0)

    q = qpos[:, 3:7]
    rel = quat.quat_mul(quat.quat_conj(q[:-1], np), q[1:], np)  # body-frame step k -> k+1
    step_omega = quat.quat_to_rotvec(rel, np) / dt  # (n-1, 3)
    ang = np.empty((n, 3))
    ang[0] = step_omega[0]
    ang[-1] = step_omega[-1]
    if n > 2:
        ang[1:-1] = 0.5 * (step_omega[:-1] + step_omega[1:])
    return np.concatenate([lin, ang, joints], axis=1)


def save_clip(path: Path, clip: ReferenceClip) -> None:
    """Validate and write ``clip`` as a compressed ``.npz``."""
    clip.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fps_hz=np.float64(clip.fps_hz),
        qpos=clip.qpos.astype(np.float64),
        qvel=clip.qvel.astype(np.float64),
        joint_names=np.array(clip.joint_names),
        site_names=np.array(clip.site_names),
        site_pos=clip.site_pos.astype(np.float64),
        licence=np.array(clip.licence),
        source=np.array(clip.source),
        foot_contact=np.asarray(clip.foot_contact, dtype=bool),
    )


def load_clip(path: Path) -> ReferenceClip:
    """Load and validate a clip written by ``save_clip``."""
    if not path.is_file():
        raise ClipError(f"reference clip not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"fps_hz", "qpos", "qvel", "joint_names", "site_names", "site_pos", "licence", "source"}
        missing = required - set(data.files)
        if missing:
            raise ClipError(f"{path}: missing fields {sorted(missing)}")
        clip = ReferenceClip(
            fps_hz=float(data["fps_hz"]),
            qpos=np.asarray(data["qpos"], dtype=np.float64),
            qvel=np.asarray(data["qvel"], dtype=np.float64),
            joint_names=tuple(str(s) for s in data["joint_names"]),
            site_names=tuple(str(s) for s in data["site_names"]),
            site_pos=np.asarray(data["site_pos"], dtype=np.float64),
            licence=str(data["licence"]),
            source=str(data["source"]),
            foot_contact=np.asarray(data["foot_contact"], dtype=bool)
            if "foot_contact" in data.files
            else np.zeros(0, dtype=bool),
        )
    clip.validate()
    return clip


def sample_clip(
    qpos: Array,
    qvel: Array,
    site_pos: Array,
    fps_hz: float,
    t_s: Array,
    xp: ModuleType,
) -> tuple[Array, Array, Array]:
    """Interpolate the clip at time ``t_s`` (s).

    Positions and velocities are linearly interpolated, the root quaternion
    with nlerp. Times before 0 return frame 0. Times after the last frame
    return the last pose with **zero** velocity, i.e. the reference holds the
    final frame (used for the post-landing hold phase).
    """
    n = qpos.shape[0]
    f = xp.clip(t_s * fps_hz, 0.0, n - 1.0)
    i0 = xp.clip(xp.floor(f).astype(xp.int32), 0, n - 2)
    alpha = f - i0
    i1 = i0 + 1

    def lerp(a: Array) -> Array:
        return a[i0] * (1.0 - alpha) + a[i1] * alpha

    root_pos = lerp(qpos[:, 0:3])
    root_quat = quat.quat_nlerp(qpos[i0, 3:7], qpos[i1, 3:7], alpha, xp)
    joints = lerp(qpos[:, ROOT_QPOS_DIM:])
    q = xp.concatenate([root_pos, root_quat, joints], axis=-1)

    past_end = t_s > (n - 1.0) / fps_hz
    v = xp.where(past_end, 0.0, lerp(qvel))
    s = lerp(site_pos)
    return q, v, s


def sample_qpos_batch(clip: ReferenceClip, times_s: np.ndarray) -> np.ndarray:
    """Reference ``qpos`` at an array of times (any shape) -> ``times.shape + (nq,)`` (numpy).

    Same interpolation as ``sample_clip`` (linear, nlerp for the root quaternion,
    clamped to the clip, holding the last frame after its end).
    """
    t = np.asarray(times_s, dtype=np.float64)
    n = clip.num_frames
    f = np.clip(t * clip.fps_hz, 0.0, n - 1.0)
    i0 = np.clip(np.floor(f).astype(np.int64), 0, n - 2)
    a = (f - i0)[..., None]
    q0, q1 = clip.qpos[i0], clip.qpos[i0 + 1]
    out = q0 * (1.0 - a) + q1 * a
    out[..., 3:7] = quat.quat_nlerp(q0[..., 3:7], q1[..., 3:7], a[..., 0], np)
    return out
