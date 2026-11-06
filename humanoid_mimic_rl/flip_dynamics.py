"""Floating-base dynamics utilities for flip references (requires ``mujoco``; no JAX).

Used by the backflip trajectory optimiser (``flip_opt``) and by the
independent feasibility checker (``check_clip``):

* ``FloatingBaseModel`` wraps a MuJoCo model with all constraints disabled
  (contacts, joint limits, friction loss), so ``mj_inverse`` returns
  ``qfrc_inverse = M qacc + c - passive``: the generalized force that
  actuators **plus** external contact wrenches must supply;
* ``foot_wrench`` splits the floating-base rows of that force into one
  wrench per foot (equal on both feet, applied at the sole centre; valid for
  the left/right-symmetric sagittal motions used here) and returns the joint
  torques that remain for the actuators;
* ``actuator_caps`` gives the torque each actuator can produce at a joint
  angle: ``min(force limit, kp * |ctrl limit - q|)`` for position servos
  (the model's ``kv`` is 0), i.e. the PD authority, per direction.

Frames / units: world frame z-up, SI units; MuJoCo free-joint conventions
(``qvel[0:3]`` world linear velocity, ``qvel[3:6]`` body angular velocity,
``qfrc[0:3]`` world force, ``qfrc[3:6]`` body-frame torque about the root).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import mujoco
import numpy as np

from humanoid_mimic_rl import quat, reference

GRAVITY_MPS2 = 9.81
COP_TOLERANCE_M = 0.005  # CoP may lie this far outside the touching sole region (numerics)


@dataclasses.dataclass(frozen=True)
class FootGeometry:
    bodies: tuple[str, ...]
    sole_centre_local_m: tuple[float, float, float]
    half_length_m: float
    half_width_m: float


class FloatingBaseModel:
    """MuJoCo model + data with constraints disabled, plus robot metadata."""

    def __init__(self, model: mujoco.MjModel, feet: FootGeometry, floor_geoms: Sequence[str]) -> None:
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("first joint must be the free joint")
        self.m = model
        self.m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        self.d = mujoco.MjData(self.m)
        self.feet = feet
        self.foot_ids = np.array([model.body(b).id for b in feet.bodies])
        self.sole_local = np.asarray(feet.sole_centre_local_m)
        hx, hy = feet.half_length_m, feet.half_width_m
        sx, sy, sz = feet.sole_centre_local_m
        self.sole_corners_local = np.array([[sx + a * hx, sy + b * hy, sz] for a in (-1, 1) for b in (-1, 1)])
        self.clear_geom_ids = np.array([model.geom(g).id for g in floor_geoms])
        self.nv = model.nv
        self.mass_kg = float(model.body_subtreemass[1])
        dof = np.array([model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in range(model.nu)])
        if not np.array_equal(dof, np.arange(6, 6 + model.nu)):
            raise ValueError("expected one actuator per joint, in joint order")
        jnt = model.actuator_trnid[:, 0]
        self.force_limit = np.max(np.abs(model.jnt_actfrcrange[jnt]), axis=1)
        if np.any(self.force_limit <= 0.0):
            raise ValueError("every actuated joint needs actuatorfrcrange")
        self.kp = model.actuator_gainprm[:, 0].copy()
        # Position servos (affine bias) are limited by PD authority; torque motors
        # (no bias) only by their force limit / control range.
        self.is_servo = model.actuator_biastype == int(mujoco.mjtBias.mjBIAS_AFFINE)
        is_motor = ~self.is_servo
        ctrl_abs = np.max(np.abs(model.actuator_ctrlrange), axis=1)
        self.force_limit = np.where(is_motor & (model.actuator_ctrllimited == 1),
                                    np.minimum(self.force_limit, ctrl_abs), self.force_limit)
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
        self.jnt_lo = model.jnt_range[1:, 0].copy()
        self.jnt_hi = model.jnt_range[1:, 1].copy()

    # ------------------------------------------------------------------
    def actuator_caps(self, q_joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(cap_pos, cap_neg)``: max positive / negative torque magnitude at ``q_joint``.

        Servos: ``min(force limit, kp * distance to the control limit)``; motors: force limit.
        """
        pos = np.minimum(self.force_limit, np.maximum(self.kp * (self.ctrl_hi - q_joint), 0.0))
        neg = np.minimum(self.force_limit, np.maximum(self.kp * (q_joint - self.ctrl_lo), 0.0))
        return np.where(self.is_servo, pos, self.force_limit), np.where(self.is_servo, neg, self.force_limit)

    def kinematics(self, qpos: np.ndarray) -> None:
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)

    def foot_pose(self, qpos: np.ndarray, foot: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """World position and rotation matrix of a foot body."""
        self.kinematics(qpos)
        b = self.foot_ids[foot]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def root_accel(self, qpos: np.ndarray, qvel: np.ndarray, qacc_joint: np.ndarray) -> np.ndarray:
        """Root acceleration with zero external wrench (flight), given joint accelerations."""
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        mujoco.mj_fwdPosition(self.m, self.d)
        mujoco.mj_fwdVelocity(self.m, self.d)
        full = self.dense_mass_matrix()
        rhs = -(self.d.qfrc_bias[:6] - self.d.qfrc_passive[:6]) - full[:6, 6:] @ qacc_joint
        return np.linalg.solve(full[:6, :6], rhs)

    def dense_mass_matrix(self) -> np.ndarray:
        """Dense joint-space inertia (after ``mj_fwdPosition``).

        MuJoCo changed the binding from ``mj_fullM(m, dst, d.qM)`` to
        ``mj_fullM(m, d, dst)`` (and renamed ``qM``); both are supported because
        playground only requires ``mujoco >= 3.6``.
        """
        full = np.zeros((self.nv, self.nv))
        if hasattr(self.d, "qM"):
            mujoco.mj_fullM(self.m, full, self.d.qM)
        else:
            mujoco.mj_fullM(self.m, self.d, full)
        return full

    def inverse(self, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray) -> np.ndarray:
        """``qfrc_inverse`` (constraints disabled)."""
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        self.d.qacc[:] = qacc
        mujoco.mj_inverse(self.m, self.d)
        return self.d.qfrc_inverse.copy()

    def foot_wrench(self, qfrc_inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-foot wrench ``[F (3), tau (3)]`` (world, about the sole centre) balancing the
        root rows, and the joint torques left for the actuators. Call after ``inverse``."""
        basis = np.zeros((self.nv, 6))
        col = np.zeros(self.nv)
        for k in range(6):
            force = np.zeros(3)
            torque = np.zeros(3)
            (force if k < 3 else torque)[k % 3] = 1.0
            col[:] = 0.0
            for b in self.foot_ids:
                point = self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.sole_local
                mujoco.mj_applyFT(self.m, self.d, force, torque, point, int(b), col)
            basis[:, k] = col
        w = np.linalg.solve(basis[:6], qfrc_inverse[:6])
        tau = qfrc_inverse[6:] - basis[6:] @ w
        return w, tau

    def sole_corner_heights(self, foot: int) -> np.ndarray:
        """Heights (m) of the four sole corners of a foot; call after kinematics."""
        b = self.foot_ids[foot]
        return (self.d.xpos[b] + self.sole_corners_local @ self.d.xmat[b].reshape(3, 3).T)[:, 2]

    def sole_centre_and_corners(self, foot: int) -> tuple[np.ndarray, np.ndarray]:
        """World sole centre (3,) and corners (4, 3) of a foot; call after kinematics."""
        b = self.foot_ids[foot]
        pos, rot = self.d.xpos[b], self.d.xmat[b].reshape(3, 3)
        return pos + rot @ self.sole_local, pos + self.sole_corners_local @ rot.T

    def body_clearance(self) -> float:
        """Lowest bounding-sphere bottom (m) of the non-foot floor colliders; after kinematics."""
        z = self.d.geom_xpos[self.clear_geom_ids, 2] - self.m.geom_rbound[self.clear_geom_ids]
        return float(np.min(z))


def ground_cop(centre: np.ndarray, wrench: np.ndarray) -> np.ndarray:
    """Centre of pressure (x, y) on the floor plane z = 0 of a wrench ``[F, tau]``
    given about ``centre`` (world). Requires ``F_z > 0``."""
    f, tau = wrench[:3], wrench[3:]
    dx = (-centre[2] * f[0] - tau[1]) / f[2]
    dy = (tau[0] - centre[2] * f[1]) / f[2]
    return np.array([centre[0] + dx, centre[1] + dy])


def cop_outside_support(cop: np.ndarray, corners: np.ndarray, touch_tol_m: float = 0.005) -> float:
    """Distance (m) by which ``cop`` lies outside the box spanned by the sole corners
    touching the floor (height < ``touch_tol_m``); 0 inside. With the heel lifted only
    the toe corners touch, so the support shrinks to the toe line."""
    touching = corners[corners[:, 2] < touch_tol_m]
    if touching.shape[0] == 0:
        return float("inf")
    lo, hi = touching[:, :2].min(axis=0), touching[:, :2].max(axis=0)
    out = np.maximum(lo - cop, 0.0) + np.maximum(cop - hi, 0.0)
    return float(np.linalg.norm(out))


@dataclasses.dataclass(frozen=True)
class DynamicsReport:
    """Inverse-dynamics feasibility numbers for a clip (see ``check_clip``)."""

    frames: int
    stance_frames: int
    flight_frames: int
    excluded_frames: int
    flight_root_force_max_n: float
    flight_root_torque_max_nm: float
    body_weight_n: float
    stance_min_normal_force_n: float
    stance_max_friction_ratio: float
    stance_max_cop_outside_m: float
    torque_ratio_max: float
    torque_ratio_by_joint: dict[str, float]
    worst_joint: str
    joints_over_limit: list[str]

    def passes(self, torque_margin: float, friction: float, root_force_tol_n: float, root_torque_tol_nm: float) -> bool:
        """All checks: flight root wrench within tolerance, feasible stance wrench, torques within margin."""
        return (
            self.flight_root_force_max_n <= root_force_tol_n
            and self.flight_root_torque_max_nm <= root_torque_tol_nm
            and self.stance_min_normal_force_n >= 0.0
            and self.stance_max_friction_ratio <= friction
            and self.stance_max_cop_outside_m <= COP_TOLERANCE_M
            and self.torque_ratio_max <= torque_margin
        )


def check_clip(
    fb: FloatingBaseModel,
    clip: reference.ReferenceClip,
    foot_contact: np.ndarray,
    joint_names: Sequence[str],
) -> DynamicsReport:
    """Inverse-dynamics feasibility of a clip, independent of how it was generated.

    Velocities and accelerations are finite differences of ``clip.qpos``
    (``reference.finite_difference_qvel`` twice). Frames whose 3-point
    stencil straddles a contact change are excluded (the true acceleration
    is discontinuous there). Flight frames must need ~zero root wrench;
    stance frames need a foot wrench with positive normal force, inside the
    friction cone and with the centre of pressure on the sole; all frames
    need joint torques within ``actuator_caps``.
    """
    qpos = clip.qpos
    qvel = reference.finite_difference_qvel(qpos, clip.fps_hz)
    qacc = np.gradient(qvel, 1.0 / clip.fps_hz, axis=0)
    contact = np.asarray(foot_contact, dtype=bool)
    n = qpos.shape[0]
    changed = np.zeros(n, dtype=bool)
    flip = np.flatnonzero(contact[1:] != contact[:-1])
    for i in flip:
        changed[max(i - 1, 0) : min(i + 3, n)] = True
    changed[[0, 1, -2, -1]] = True  # one-sided differences at the clip ends
    root_f, root_t, fz_min, fric, cop_out = [0.0], [0.0], [np.inf], [0.0], [0.0]
    ratio = np.zeros(fb.m.nu)
    for k in range(n):
        if changed[k]:
            continue
        f = fb.inverse(qpos[k], qvel[k], qacc[k])
        cap_pos, cap_neg = fb.actuator_caps(qpos[k, 7:])
        if contact[k]:
            w, tau = fb.foot_wrench(f)
            fz = w[2]
            fz_min.append(fz)
            fric.append(np.hypot(w[0], w[1]) / max(fz, 1e-9))
            centre, corners = fb.sole_centre_and_corners(0)
            cop_out.append(cop_outside_support(ground_cop(centre, w), corners) if fz > 0 else float("inf"))
        else:
            tau = f[6:]
            root_f.append(float(np.linalg.norm(f[0:3])))
            root_t.append(float(np.linalg.norm(f[3:6])))
        cap = np.where(tau >= 0.0, cap_pos, cap_neg)
        ratio = np.maximum(ratio, np.abs(tau) / np.maximum(cap, 1e-9))
    by_joint = {name: float(r) for name, r in zip(joint_names, ratio)}
    worst = max(by_joint, key=by_joint.get)
    return DynamicsReport(
        frames=n,
        stance_frames=int(np.sum(contact & ~changed)),
        flight_frames=int(np.sum(~contact & ~changed)),
        excluded_frames=int(np.sum(changed)),
        flight_root_force_max_n=float(np.max(root_f)),
        flight_root_torque_max_nm=float(np.max(root_t)),
        body_weight_n=fb.mass_kg * GRAVITY_MPS2,
        stance_min_normal_force_n=float(np.min(fz_min)) if len(fz_min) > 1 else float("nan"),
        stance_max_friction_ratio=float(np.max(fric)),
        stance_max_cop_outside_m=float(np.max(cop_out)),
        torque_ratio_max=float(np.max(ratio)),
        torque_ratio_by_joint=by_joint,
        worst_joint=worst,
        joints_over_limit=[j for j, r in by_joint.items() if r > 1.0],
    )


def pitch_of(rot: np.ndarray) -> float:
    """Rotation angle about world y of a rotation matrix that is (close to) a pure pitch."""
    return float(np.arctan2(rot[0, 2], rot[0, 0]))


def mat_to_quat(rot: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, rot.reshape(-1))
    return out


def body_rates(q_minus: np.ndarray, q0: np.ndarray, q_plus: np.ndarray, eps_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame angular velocity and acceleration from three orientations ``eps_s`` apart."""
    r_p = quat.quat_to_rotvec(quat.quat_mul(quat.quat_conj(q0, np), q_plus, np), np)
    r_m = quat.quat_to_rotvec(quat.quat_mul(quat.quat_conj(q0, np), q_minus, np), np)
    return (r_p - r_m) / (2.0 * eps_s), (r_p + r_m) / eps_s**2
