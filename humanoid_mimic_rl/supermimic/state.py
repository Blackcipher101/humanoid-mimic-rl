"""World-model state representation (backend-agnostic, ``xp`` = numpy or jax.numpy).

The simulator state of a floating-base robot is ``qpos = [p (3, world m),
q (4, wxyz world<-body), joints (n, rad)]`` and ``qvel = [v (3, world m/s),
w (3, body rad/s), joint rates (n, rad/s)]`` (MuJoCo free-joint conventions).

The world model sees a representation that is invariant to horizontal
translation and yaw, as the physics on flat ground is:

* **input features** ``[z, g_body (3), joints, v_body (3), w_body (3),
  joint rates, u_rel]`` where ``g_body = R^T (0, 0, -1)`` is gravity in the
  body frame, ``v_body = R^T v`` and ``u_rel = joint targets - joints`` (the
  position-actuator error that drives the torques);
* **outputs** (per control step) ``[dp_body (3), dtheta_body (3),
  d_joints, v'_body' (3), w' (3), joint rates']`` where ``dp_body = R^T (p' - p)``,
  ``dtheta_body = log(q^-1 q')`` and ``v'_body' = R'^T v'`` (next body frame).

``integrate`` inverts ``targets_from_transition`` exactly (up to quaternion
sign), and ``state_error`` measures prediction error in the same 3+3+n+3+3+n
layout as the outputs so a single per-dimension scale normalizes both.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from humanoid_mimic_rl import quat

Array = Any
GRAVITY_DIR = (0.0, 0.0, -1.0)


def num_features(n_joints: int) -> int:
    return 1 + 3 + n_joints + 3 + 3 + n_joints + n_joints


def num_outputs(n_joints: int) -> int:
    return 3 + 3 + n_joints + 3 + 3 + n_joints


def _inv_rotate(q: Array, v: Array, xp: ModuleType) -> Array:
    """``R(q)^T v``: world-frame vector into the body frame."""
    return quat.quat_rotate(quat.quat_conj(q, xp), v, xp)


def gravity_body(q: Array, xp: ModuleType) -> Array:
    g = xp.zeros(q.shape[:-1] + (3,)) + xp.asarray(GRAVITY_DIR)
    return _inv_rotate(q, g, xp)


def features(qpos: Array, qvel: Array, u_rel: Array, xp: ModuleType) -> Array:
    """World-model input features (see module docstring)."""
    q = qpos[..., 3:7]
    return xp.concatenate(
        [
            qpos[..., 2:3],
            gravity_body(q, xp),
            qpos[..., 7:],
            _inv_rotate(q, qvel[..., 0:3], xp),
            qvel[..., 3:6],
            qvel[..., 6:],
            u_rel,
        ],
        axis=-1,
    )


def targets_from_transition(
    qpos: Array, qvel: Array, qpos_next: Array, qvel_next: Array, xp: ModuleType
) -> Array:
    """World-model regression target for the transition ``(qpos, qvel) -> (qpos', qvel')``."""
    q, q_next = qpos[..., 3:7], qpos_next[..., 3:7]
    dp_body = _inv_rotate(q, qpos_next[..., 0:3] - qpos[..., 0:3], xp)
    dtheta = quat.quat_to_rotvec(quat.quat_mul(quat.quat_conj(q, xp), q_next, xp), xp)
    return xp.concatenate(
        [
            dp_body,
            dtheta,
            qpos_next[..., 7:] - qpos[..., 7:],
            _inv_rotate(q_next, qvel_next[..., 0:3], xp),
            qvel_next[..., 3:6],
            qvel_next[..., 6:],
        ],
        axis=-1,
    )


def integrate(qpos: Array, qvel: Array, out: Array, xp: ModuleType) -> tuple[Array, Array]:
    """Apply a world-model output to ``(qpos, qvel)``; returns ``(qpos', qvel')``."""
    n = qpos.shape[-1] - 7
    q = qpos[..., 3:7]
    dp_body, dtheta = out[..., 0:3], out[..., 3:6]
    d_joints = out[..., 6 : 6 + n]
    v_body_next = out[..., 6 + n : 9 + n]
    w_next = out[..., 9 + n : 12 + n]
    qd_next = out[..., 12 + n : 12 + 2 * n]
    p_next = qpos[..., 0:3] + quat.quat_rotate(q, dp_body, xp)
    q_next = quat.quat_normalize(quat.quat_mul(q, quat.rotvec_to_quat(dtheta, xp), xp), xp)
    qpos_next = xp.concatenate([p_next, q_next, qpos[..., 7:] + d_joints], axis=-1)
    qvel_next = xp.concatenate([quat.quat_rotate(q_next, v_body_next, xp), w_next, qd_next], axis=-1)
    return qpos_next, qvel_next


def state_error(
    qpos_hat: Array, qvel_hat: Array, qpos: Array, qvel: Array, xp: ModuleType
) -> Array:
    """Prediction error in the output layout: position (world m), orientation
    (body rotation vector rad), joints, linear velocity (body frame of the true
    state), angular velocity, joint rates."""
    q = qpos[..., 3:7]
    rot_err = quat.quat_to_rotvec(quat.quat_mul(quat.quat_conj(q, xp), qpos_hat[..., 3:7], xp), xp)
    return xp.concatenate(
        [
            _inv_rotate(q, qpos_hat[..., 0:3] - qpos[..., 0:3], xp),
            rot_err,
            qpos_hat[..., 7:] - qpos[..., 7:],
            _inv_rotate(q, qvel_hat[..., 0:3] - qvel[..., 0:3], xp),
            qvel_hat[..., 3:6] - qvel[..., 3:6],
            qvel_hat[..., 6:] - qvel[..., 6:],
        ],
        axis=-1,
    )


def phase_features(t_s: Array, period_s: float, xp: ModuleType) -> Array:
    """Same as ``envs.backflip.phase_features``: ``[phase, sin 2 pi phase, cos 2 pi phase]``."""
    phase = xp.clip(t_s / period_s, 0.0, 1.0)
    return xp.stack([phase, xp.sin(2.0 * xp.pi * phase), xp.cos(2.0 * xp.pi * phase)], axis=-1)


def actor_obs(
    qpos: Array,
    qvel: Array,
    last_act: Array,
    t_s: Array,
    default_pose: Array,
    period_s: float,
    xp: ModuleType,
) -> Array:
    """Noise-free actor observation, identical in layout to
    ``T1FullCollisionEnv.proprio_obs(...)["state"]`` with phase features:
    ``[gyro (= body angular velocity), g_body, joints - default, joint rates,
    last action, phase (3)]``. Valid because the T1 IMU site sits at the
    trunk origin with identity orientation (checked at runtime by the trainer).
    """
    return xp.concatenate(
        [
            qvel[..., 3:6],
            gravity_body(qpos[..., 3:7], xp),
            qpos[..., 7:] - default_pose,
            qvel[..., 6:],
            last_act,
            phase_features(t_s, period_s, xp),
        ],
        axis=-1,
    )


def critic_features(qpos: Array, qvel: Array, last_act: Array, t_s: Array, period_s: float, xp: ModuleType) -> Array:
    """Privileged critic input: root height, gravity, body velocities, joints, last action, phase."""
    q = qpos[..., 3:7]
    return xp.concatenate(
        [
            qpos[..., 2:3],
            gravity_body(q, xp),
            _inv_rotate(q, qvel[..., 0:3], xp),
            qvel[..., 3:6],
            qpos[..., 7:],
            qvel[..., 6:],
            last_act,
            phase_features(t_s, period_s, xp),
        ],
        axis=-1,
    )


def window_valid_mask(done: Array, xp: ModuleType) -> Array:
    """``(B, k)`` mask of usable predictions in a replay window.

    ``done[:, j]`` = transition ``j`` of the window ended an episode, so the
    stored row ``j + 1`` is a reset state, not its successor. Prediction ``j``
    (of row ``j + 1``) is valid only if none of transitions ``0..j`` ended.
    """
    return xp.cumsum(done.astype(xp.int32), axis=1) == 0


def ring_rows(ptr: Array, size: Array, capacity: int, offsets: Array, xp: ModuleType) -> Array:
    """Physical rows of logical (oldest-first) positions ``offsets`` in a ring buffer
    whose next write goes to ``ptr`` and which holds ``size`` rows."""
    return xp.mod(ptr - size + offsets, capacity)
