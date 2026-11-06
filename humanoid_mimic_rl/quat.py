"""Backend-agnostic quaternion math.

Conventions (identical to MuJoCo):
  * quaternions are ``(w, x, y, z)`` along the last axis, Hamilton product;
  * ``quat_rotate(q, v)`` maps a vector expressed in the body frame into the
    parent (world) frame, i.e. ``v_world = R(q) v_body``;
  * rotation vectors are axis * angle in radians.

Every function takes the array namespace ``xp`` explicitly (``numpy`` or
``jax.numpy``), so the same code runs inside jitted MJX environments and in
pure-numpy unit tests / offline processing. Only operations that exist with
identical semantics in both namespaces are used.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

Array = Any  # numpy.ndarray or jax.Array; kept loose to avoid importing jax.

# Below this rotation-vector norm (rad) the small-angle series is used.
_SMALL_ANGLE_RAD = 1e-6


def quat_mul(a: Array, b: Array, xp: ModuleType) -> Array:
    """Hamilton product ``a * b`` (apply ``b`` first, then ``a``)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return xp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: Array, xp: ModuleType) -> Array:
    """Conjugate (inverse for unit quaternions)."""
    return xp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_normalize(q: Array, xp: ModuleType) -> Array:
    """Normalize to unit length. Zero quaternions are returned as identity."""
    norm = xp.sqrt(xp.sum(q * q, axis=-1, keepdims=True))
    identity = xp.zeros_like(q) + xp.asarray([1.0, 0.0, 0.0, 0.0], dtype=q.dtype)
    safe = xp.where(norm > 0.0, norm, 1.0)
    return xp.where(norm > 0.0, q / safe, identity)


def quat_rotate(q: Array, v: Array, xp: ModuleType) -> Array:
    """Rotate vector ``v`` (body frame) by unit quaternion ``q`` into the parent frame."""
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * xp.cross(u, v)
    return v + w * t + xp.cross(u, t)


def quat_from_axis_angle(axis: Array, angle_rad: Array, xp: ModuleType) -> Array:
    """Quaternion for a rotation of ``angle_rad`` about unit vector ``axis``."""
    half = 0.5 * xp.asarray(angle_rad)[..., None]
    return xp.concatenate([xp.cos(half), xp.sin(half) * axis], axis=-1)


def rotvec_to_quat(rotvec: Array, xp: ModuleType) -> Array:
    """Exponential map: rotation vector (rad) -> unit quaternion."""
    angle = xp.sqrt(xp.sum(rotvec * rotvec, axis=-1, keepdims=True))
    small = angle < _SMALL_ANGLE_RAD
    safe_angle = xp.where(small, 1.0, angle)
    # sin(a/2)/a -> 1/2 as a -> 0.
    scale = xp.where(small, 0.5, xp.sin(0.5 * safe_angle) / safe_angle)
    return xp.concatenate([xp.cos(0.5 * angle), rotvec * scale], axis=-1)


def quat_to_rotvec(q: Array, xp: ModuleType) -> Array:
    """Logarithmic map: unit quaternion -> rotation vector with angle in [0, pi].

    ``q`` and ``-q`` encode the same rotation; the shortest rotation is returned.
    """
    sign = xp.where(q[..., :1] < 0.0, -1.0, 1.0)
    q = q * sign
    v = q[..., 1:]
    vnorm = xp.sqrt(xp.sum(v * v, axis=-1, keepdims=True))
    angle = 2.0 * xp.arctan2(vnorm, q[..., :1])
    small = vnorm < _SMALL_ANGLE_RAD
    safe_vnorm = xp.where(small, 1.0, vnorm)
    # angle / |v| -> 2 / w as |v| -> 0 (w ~ 1 after sign fix).
    scale = xp.where(small, 2.0, angle / safe_vnorm)
    return v * scale


def quat_angle_between(q1: Array, q2: Array, xp: ModuleType) -> Array:
    """Geodesic angle (rad, in [0, pi]) of the rotation taking ``q1`` to ``q2``."""
    rel = quat_mul(quat_conj(q1, xp), q2, xp)
    rv = quat_to_rotvec(rel, xp)
    return xp.sqrt(xp.sum(rv * rv, axis=-1))


def quat_nlerp(q0: Array, q1: Array, alpha: Array, xp: ModuleType) -> Array:
    """Normalized linear interpolation along the shortest arc.

    Equivalent to slerp to within ~1e-3 rad for the sub-0.2 rad steps between
    consecutive reference frames at >= 50 Hz, and cheaper and branch-free.
    """
    dot = xp.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = xp.where(dot < 0.0, -q1, q1)
    alpha = xp.asarray(alpha)[..., None]
    return quat_normalize((1.0 - alpha) * q0 + alpha * q1, xp)
