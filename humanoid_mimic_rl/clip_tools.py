"""Helpers for clips recorded from simulation (numpy + mujoco, no JAX)."""

from __future__ import annotations

import copy

import mujoco
import numpy as np

from humanoid_mimic_rl import quat, t1_model

CONTACT_TOL_M = 0.005  # a foot sphere this close to the floor counts as contact


def yaw_align(qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate/rotate so frame 0 has root (x, y) = 0 and heading +x (yaw about world z)."""
    q0 = qpos[0, 3:7]
    yaw = np.arctan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))
    z = np.array([0.0, 0.0, 1.0])
    q_inv = quat.quat_from_axis_angle(z, -yaw, np)
    qpos, qvel = qpos.copy(), qvel.copy()
    offset = np.array([qpos[0, 0], qpos[0, 1], 0.0])
    qpos[:, 0:3] = quat.quat_rotate(np.tile(q_inv, (len(qpos), 1)), qpos[:, 0:3] - offset, np)
    qpos[:, 3:7] = quat.quat_normalize(quat.quat_mul(np.tile(q_inv, (len(qpos), 1)), qpos[:, 3:7], np), np)
    qvel[:, 0:3] = quat.quat_rotate(np.tile(q_inv, (len(qvel), 1)), qvel[:, 0:3], np)  # world linear
    return qpos, qvel  # body-frame angular velocity and joint rates are unchanged


def sanity_inverse_dynamics(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, fps: float) -> dict[str, object]:
    """Approximate actuator torque demand (min-norm foot-wrench split) vs joint force limits.

    Works on a copy of ``model`` with constraints disabled (``qfrc_inverse`` is then
    what actuators plus foot contacts must supply). The split of the floating-base
    wrench between two feet in contact is not unique (minimum-norm is used), so
    the torque numbers are indicative only.
    """
    m = copy.deepcopy(model)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    d = mujoco.MjData(m)
    qacc = np.gradient(qvel, 1.0 / fps, axis=0)
    feet = [m.body("left_foot_link").id, m.body("right_foot_link").id]
    spheres = [[m.geom(f"{side}_foot_{i}").id for i in range(1, 5)] for side in ("left", "right")]
    limit = np.max(np.abs(m.jnt_actfrcrange[1:]), axis=1)
    ratio = np.zeros(m.nu)
    airborne_root_force = 0.0
    col = np.zeros(m.nv)
    for k in range(1, len(qpos) - 1):
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos[k], qvel[k], qacc[k]
        mujoco.mj_inverse(m, d)
        f = d.qfrc_inverse.copy()
        touching = [i for i in range(2) if np.min(d.geom_xpos[spheres[i], 2] - 0.01) < CONTACT_TOL_M]
        if not touching:
            airborne_root_force = max(airborne_root_force, float(np.linalg.norm(f[:3])))
            tau = f[6:]
        else:
            basis = []
            for i in touching:
                for axis in range(6):
                    force, torque = np.zeros(3), np.zeros(3)
                    (force if axis < 3 else torque)[axis % 3] = 1.0
                    col[:] = 0.0
                    mujoco.mj_applyFT(m, d, force, torque, d.xpos[feet[i]], feet[i], col)
                    basis.append(col.copy())
            a = np.array(basis).T
            w = np.linalg.lstsq(a[:6], f[:6], rcond=None)[0]
            tau = f[6:] - a[6:] @ w
        ratio = np.maximum(ratio, np.abs(tau) / limit)
    worst = int(np.argmax(ratio))
    return {
        "approx_max_torque_over_force_limit": float(ratio[worst]),
        "approx_worst_joint": t1_model.T1_JOINT_NAMES[worst],
        "max_root_force_when_no_foot_touches_n": airborne_root_force,
    }
