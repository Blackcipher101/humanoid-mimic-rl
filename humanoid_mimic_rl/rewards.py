"""Backend-agnostic reward kernels.

All kernels are pure functions of arrays and take the array namespace ``xp``
(``numpy`` or ``jax.numpy``) explicitly, so the exact code used by the jitted
environments is unit-tested with numpy. Positive "reward" kernels return
values in [0, 1]; "cost" kernels return non-negative values that the
environment multiplies by a negative scale.

Units are SI throughout: metres, seconds, radians, newton-metres.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from humanoid_mimic_rl import quat

Array = Any


# ---------------------------------------------------------------------------
# Generic shaping primitives.
# ---------------------------------------------------------------------------


def exp_kernel(sq_error: Array, gain: float, xp: ModuleType) -> Array:
    """``exp(-gain * sq_error)``; 1 at zero error, decays with squared error."""
    return xp.exp(-gain * sq_error)


def time_window(t_s: Array, start_s: float, end_s: float, xp: ModuleType) -> Array:
    """1.0 while ``start_s <= t_s < end_s`` else 0.0."""
    return xp.where((t_s >= start_s) & (t_s < end_s), 1.0, 0.0)


def ramp_reward(value: Array, target: float, xp: ModuleType) -> Array:
    """Linear reward ``clip(value / target, 0, 1)`` for a positive ``target``."""
    return xp.clip(value / target, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Posture / balance terms.
# ---------------------------------------------------------------------------


def upright_reward(up_z: Array, xp: ModuleType) -> Array:
    """Smooth uprightness reward from the world-z component of the torso z-axis.

    ``up_z`` = 1 upright, 0 lying horizontally, -1 upside-down. Returns
    ``((1 + up_z) / 2)^2`` in [0, 1].
    """
    return xp.square(0.5 * (1.0 + xp.clip(up_z, -1.0, 1.0)))


def height_reward(height_m: Array, target_height_m: float, xp: ModuleType) -> Array:
    """``clip(height / target, 0, 1)``; saturates once the target is reached."""
    return xp.clip(height_m / target_height_m, 0.0, 1.0)


def tall_upright_reward(up_z: Array, height_m: Array, target_height_m: float, xp: ModuleType) -> Array:
    """Uprightness that only pays when the torso is also high: ``upright(up_z) * clip(h / h_target)^2``.

    Rewarding uprightness and height separately lets a policy collect most of the
    reward by sitting up straight on the floor (a local optimum seen in the first
    full get-up run); the product removes that plateau.
    """
    frac = xp.clip(height_m / target_height_m, 0.0, 1.0)
    return upright_reward(up_z, xp) * xp.square(frac)


def feet_under_torso_reward(torso_xy: Array, feet_xy: Array, length_scale_m: float, xp: ModuleType) -> Array:
    """``exp(-(d / L)^2)`` with ``d`` the horizontal distance from the torso to the feet midpoint.

    Shapes the sit -> squat transition: feet must come under the body before the
    robot can push itself up. ``feet_xy`` has shape ``(..., 2, 2)`` (left/right, xy).
    """
    mid = 0.5 * (feet_xy[..., 0, :] + feet_xy[..., 1, :])
    d2 = xp.sum(xp.square(mid - torso_xy), axis=-1)
    return xp.exp(-d2 / (length_scale_m * length_scale_m))


def standing_gate(
    up_z: Array,
    height_m: Array,
    min_up_z: float,
    min_height_m: float,
    xp: ModuleType,
) -> Array:
    """1.0 when the torso is upright and high enough, else 0.0."""
    return xp.where((up_z >= min_up_z) & (height_m >= min_height_m), 1.0, 0.0)


def posture_reward(
    joint_pos_rad: Array,
    target_pos_rad: Array,
    weights: Array,
    gain: float,
    xp: ModuleType,
) -> Array:
    """``exp(-gain * sum_i w_i (q_i - q*_i)^2)``."""
    sq = xp.sum(weights * xp.square(joint_pos_rad - target_pos_rad), axis=-1)
    return exp_kernel(sq, gain, xp)


# ---------------------------------------------------------------------------
# Regularisation costs (non-negative).
# ---------------------------------------------------------------------------


def action_rate_cost(
    action: Array, last_action: Array, last_last_action: Array, xp: ModuleType
) -> Array:
    """First plus second finite difference of the normalized action."""
    first = xp.sum(xp.square(action - last_action), axis=-1)
    second = xp.sum(xp.square(action - 2.0 * last_action + last_last_action), axis=-1)
    return first + second


def torque_cost(torque_nm: Array, xp: ModuleType) -> Array:
    """Sum of squared actuator torques (N^2 m^2)."""
    return xp.sum(xp.square(torque_nm), axis=-1)


def joint_limit_cost(
    joint_pos_rad: Array, soft_lower_rad: Array, soft_upper_rad: Array, xp: ModuleType
) -> Array:
    """Total violation (rad) of the soft joint limits."""
    below = xp.clip(soft_lower_rad - joint_pos_rad, 0.0, None)
    above = xp.clip(joint_pos_rad - soft_upper_rad, 0.0, None)
    return xp.sum(below + above, axis=-1)


def joint_velocity_excess_cost(
    joint_vel_radps: Array, max_vel_radps: float, xp: ModuleType
) -> Array:
    """Sum of squared joint speed in excess of ``max_vel_radps``."""
    excess = xp.clip(xp.abs(joint_vel_radps) - max_vel_radps, 0.0, None)
    return xp.sum(xp.square(excess), axis=-1)


# ---------------------------------------------------------------------------
# Backflip phase terms.
# ---------------------------------------------------------------------------


def backward_pitch_rate(angvel_body_radps: Array) -> Array:
    """Backward somersault rate (rad/s, positive when rotating backwards).

    For a body whose x-axis points forward and z-axis up, a backflip is a
    negative rotation about the body y-axis (head moves towards -x), so the
    backward rate is ``-omega_y`` with ``omega`` expressed in the body frame
    (MuJoCo free-joint ``qvel[3:6]`` convention).
    """
    return -angvel_body_radps[..., 1]


def pitch_quat(pitch_rad: Array, xp: ModuleType) -> Array:
    """World orientation of a body pitched by ``pitch_rad`` about world y (yaw = 0)."""
    axis = xp.asarray([0.0, 1.0, 0.0])
    return quat.quat_from_axis_angle(axis, pitch_rad, xp)


def orientation_tracking_reward(
    root_quat: Array, ref_quat: Array, gain: float, xp: ModuleType
) -> Array:
    """``exp(-gain * angle^2)`` with ``angle`` the geodesic root orientation error."""
    angle = quat.quat_angle_between(root_quat, ref_quat, xp)
    return exp_kernel(xp.square(angle), gain, xp)


def landing_reward(
    up_z: Array,
    height_m: Array,
    feet_in_contact: Array,
    min_up_z: float,
    min_height_m: float,
    xp: ModuleType,
) -> Array:
    """Upright, high-enough torso with at least one foot on the ground.

    ``feet_in_contact`` is a boolean array with the feet on the last axis.
    Returns ``upright_reward * gate`` in [0, 1].
    """
    any_foot = xp.any(feet_in_contact, axis=-1)
    gate = standing_gate(up_z, height_m, min_up_z, min_height_m, xp) * xp.where(
        any_foot, 1.0, 0.0
    )
    return upright_reward(up_z, xp) * gate


# ---------------------------------------------------------------------------
# DeepMimic tracking terms (Peng et al. 2018, Sec. 5).
# ---------------------------------------------------------------------------


def deepmimic_terms(
    joint_pos_rad: Array,
    ref_joint_pos_rad: Array,
    joint_vel_radps: Array,
    ref_joint_vel_radps: Array,
    root_quat: Array,
    ref_root_quat: Array,
    root_pos_m: Array,
    ref_root_pos_m: Array,
    ee_pos_m: Array,
    ref_ee_pos_m: Array,
    gains: dict[str, float],
    xp: ModuleType,
) -> dict[str, Array]:
    """Individual DeepMimic imitation terms, each in [0, 1].

    * ``pose``: joint angles plus root orientation error (rad^2);
    * ``velocity``: joint velocities (rad^2/s^2);
    * ``end_effector``: end-effector positions relative to the root, world
      axes (m^2); ``ee_pos_m`` has shape ``(..., n_ee, 3)``;
    * ``root``: root position (m^2). DeepMimic uses the centre of mass; the
      root is used here because procedurally generated clips do not carry a
      physically consistent CoM. This is a documented deviation.

    ``gains`` must contain keys ``pose``, ``velocity``, ``end_effector``, ``root``.
    """
    root_angle = quat.quat_angle_between(root_quat, ref_root_quat, xp)
    pose_sq = xp.sum(xp.square(joint_pos_rad - ref_joint_pos_rad), axis=-1) + xp.square(
        root_angle
    )
    vel_sq = xp.sum(xp.square(joint_vel_radps - ref_joint_vel_radps), axis=-1)
    ee_rel = ee_pos_m - root_pos_m[..., None, :]
    ref_ee_rel = ref_ee_pos_m - ref_root_pos_m[..., None, :]
    ee_sq = xp.sum(xp.square(ee_rel - ref_ee_rel), axis=(-2, -1))
    root_sq = xp.sum(xp.square(root_pos_m - ref_root_pos_m), axis=-1)
    return {
        "pose": exp_kernel(pose_sq, gains["pose"], xp),
        "velocity": exp_kernel(vel_sq, gains["velocity"], xp),
        "end_effector": exp_kernel(ee_sq, gains["end_effector"], xp),
        "root": exp_kernel(root_sq, gains["root"], xp),
    }


def tracking_failed(
    root_pos_m: Array,
    ref_root_pos_m: Array,
    root_quat: Array,
    ref_root_quat: Array,
    max_pos_err_m: float,
    max_angle_err_rad: float,
    xp: ModuleType,
) -> Array:
    """Early-termination test: root drifted too far from the reference."""
    pos_err = xp.sqrt(xp.sum(xp.square(root_pos_m - ref_root_pos_m), axis=-1))
    ang_err = quat.quat_angle_between(root_quat, ref_root_quat, xp)
    return (pos_err > max_pos_err_m) | (ang_err > max_angle_err_rad)


def height_tilt_failed(
    root_pos_m: Array,
    ref_root_pos_m: Array,
    root_quat: Array,
    ref_root_quat: Array,
    max_height_err_m: float,
    max_tilt_err: float,
    xp: ModuleType,
) -> Array:
    """SuperMimic (App. A) termination: root height error or tilt error too large.

    Root xy position and yaw are deliberately ignored (the actor cannot observe
    them). ``tilt_err = |g_z - g_z_ref|`` where ``g_z`` is the z component of the
    world up direction expressed in the body frame (1 upright, -1 inverted; the
    gravity direction's z component is its negative, so the difference is the
    same). Quaternions are (w, x, y, z). ``tilt_err`` lies in [0, 2]; for a pure
    tilt from an upright reference it equals ``1 - cos(tilt)``.
    """
    z_axis = xp.asarray([0.0, 0.0, 1.0])
    up_z = quat.quat_rotate(quat.quat_conj(root_quat, xp), z_axis, xp)[..., 2]
    ref_up_z = quat.quat_rotate(quat.quat_conj(ref_root_quat, xp), z_axis, xp)[..., 2]
    height_err = xp.abs(root_pos_m[..., 2] - ref_root_pos_m[..., 2])
    return (height_err > max_height_err_m) | (xp.abs(up_z - ref_up_z) > max_tilt_err)
