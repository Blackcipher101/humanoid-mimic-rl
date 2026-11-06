"""Hand-authored keyframe motions and shape-preserving interpolation (numpy only).

A keyframe motion is a list of ``Keyframe`` (time, joint-angle overrides on
top of the home pose, root pitch, foot-contact flag). Joint angles and root
pitch are interpolated with monotone piecewise-cubic Hermite splines
(Fritsch-Carlson / PCHIP), which pass through every keyframe, are C1, and
never overshoot between keyframes, so interpolated joints stay inside the
range spanned by their keyframes (and therefore inside joint limits when the
keyframes are).

The backflip keyframes below were authored for this repository for the
Booster T1 joint conventions (no third-party motion data is involved):

* hip/knee/ankle/shoulder *pitch* joints rotate about the body y-axis; with
  the leg hanging along -z a negative hip pitch swings the thigh forward
  (flexion), a positive knee pitch bends the knee, a positive ankle pitch
  points the toes down, a negative shoulder pitch raises the arm forwards /
  overhead;
* the root pitch is a rotation about world y; negative = rotating backwards
  (head towards -x), so a full backflip goes from 0 to -2*pi.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class Keyframe:
    """One keyframe.

    Attributes:
      time_s: keyframe time (s), strictly increasing within a motion.
      joints_rad: joint-name -> angle (rad) overrides; others use the home pose.
      pitch_rad: root pitch about world y (rad); negative = backwards.
      in_contact: whether the feet are on the ground at this keyframe. Contact
        between two consecutive keyframes holds only if both are in contact.
    """

    time_s: float
    joints_rad: Mapping[str, float]
    pitch_rad: float
    in_contact: bool


@dataclasses.dataclass(frozen=True)
class KeyframeTrajectory:
    """Keyframes interpolated on a uniform time grid."""

    times_s: np.ndarray  # (T,)
    joint_pos_rad: np.ndarray  # (T, n_joints)
    pitch_rad: np.ndarray  # (T,)
    in_contact: np.ndarray  # (T,) bool


@dataclasses.dataclass(frozen=True)
class FlightPhase:
    """Ballistic phase derived from the contact flags of a keyframe motion."""

    takeoff_s: float
    landing_s: float

    @property
    def duration_s(self) -> float:
        return self.landing_s - self.takeoff_s


def _symmetric(base: str, value: float) -> dict[str, float]:
    """Same angle on the left and right joint ``{Left,Right}_<base>``."""
    return {f"Left_{base}": value, f"Right_{base}": value}


def _legs(hip_pitch: float, knee: float, ankle_pitch: float) -> dict[str, float]:
    return {
        **_symmetric("Hip_Pitch", hip_pitch),
        **_symmetric("Knee_Pitch", knee),
        **_symmetric("Ankle_Pitch", ankle_pitch),
    }


def _arms(shoulder_pitch: float) -> dict[str, float]:
    return _symmetric("Shoulder_Pitch", shoulder_pitch)


TWO_PI = 2.0 * math.pi


def t1_backflip_keyframes() -> list[Keyframe]:
    """Standing backflip for the Booster T1 (authored for this repository).

    Timeline: stand (0-0.3 s), countermovement crouch with arms back
    (-> 0.70 s), explosive extension with arm swing; take-off at 0.90 s is
    placed *before* full extension (0.95 s, airborne) so joint and CoM
    velocities are non-zero at take-off; tucked rotation (flight 0.52 s), open
    up and land (1.42 s), absorb, and return to the home pose, held until 2.8 s.
    """
    return [
        Keyframe(0.00, {}, 0.0, True),
        Keyframe(0.30, {}, 0.0, True),
        Keyframe(0.70, {**_legs(-0.90, 1.60, -0.85), **_arms(0.80)}, 0.15, True),
        Keyframe(0.90, {**_legs(-0.15, 0.45, 0.20), **_arms(-2.40)}, 0.05, True),
        Keyframe(0.95, {**_legs(-0.20, 0.10, 0.30), **_arms(-2.80)}, -0.60, False),
        Keyframe(1.05, {**_legs(-1.50, 2.10, -0.30), **_arms(-1.00)}, -1.70, False),
        Keyframe(1.18, {**_legs(-1.70, 2.30, -0.30), **_arms(-1.00)}, -3.50, False),
        Keyframe(1.32, {**_legs(-0.90, 1.00, -0.40), **_arms(-1.20)}, -5.30, False),
        Keyframe(1.42, {**_legs(-0.60, 1.00, -0.40), **_arms(-1.00)}, -TWO_PI + 0.30, True),
        Keyframe(1.65, {**_legs(-1.00, 1.70, -0.75), **_arms(-0.60)}, -TWO_PI + 0.30, True),
        Keyframe(2.30, {}, -TWO_PI, True),
        Keyframe(2.80, {}, -TWO_PI, True),
    ]


def validate_keyframes(
    keyframes: Sequence[Keyframe],
    joint_names: Sequence[str],
    joint_ranges_rad: np.ndarray,
) -> None:
    """Raise ``ValueError`` for unknown joints, out-of-range angles or bad timing."""
    if len(keyframes) < 2:
        raise ValueError("need at least two keyframes")
    times = np.array([k.time_s for k in keyframes])
    if times[0] != 0.0:
        raise ValueError("first keyframe must be at t = 0")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("keyframe times must be strictly increasing")
    names = set(joint_names)
    for k in keyframes:
        for name, value in k.joints_rad.items():
            if name not in names:
                raise ValueError(f"keyframe at {k.time_s} s: unknown joint {name!r}")
            lo, hi = joint_ranges_rad[list(joint_names).index(name)]
            if not lo <= value <= hi:
                raise ValueError(
                    f"keyframe at {k.time_s} s: {name}={value} outside [{lo}, {hi}]"
                )


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone slopes for knots ``x`` (strictly increasing).

    ``y`` may have trailing channel dimensions: shape ``(n, ...)``.
    """
    h = np.diff(x)
    shape = (-1,) + (1,) * (y.ndim - 1)
    delta = np.diff(y, axis=0) / h.reshape(shape)
    m = np.zeros_like(y, dtype=np.float64)
    if len(x) == 2:
        m[0] = m[1] = delta[0]
        return m
    h0 = h[:-1].reshape(shape)
    h1 = h[1:].reshape(shape)
    d0, d1 = delta[:-1], delta[1:]
    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0
    same_sign = (d0 * d1) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / d0 + w2 / d1)
    m[1:-1] = np.where(same_sign, harmonic, 0.0)
    # Non-centred, shape-preserving end slopes (Moler, "Numerical Computing with MATLAB").
    m[0] = _end_slope(h[0], h[1], delta[0], delta[1])
    m[-1] = _end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return m


def _end_slope(h0: float, h1: float, d0: np.ndarray, d1: np.ndarray) -> np.ndarray:
    s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    s = np.where(np.sign(s) != np.sign(d0), 0.0, s)
    return np.where((np.sign(d0) != np.sign(d1)) & (np.abs(s) > np.abs(3.0 * d0)), 3.0 * d0, s)


def pchip_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate the monotone cubic Hermite interpolant at ``xq`` (clamped to [x0, xn])."""
    m = pchip_slopes(x, y)
    xq = np.clip(xq, x[0], x[-1])
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    s = (xq - x[i]) / h
    shape = (-1,) + (1,) * (y.ndim - 1)
    s = s.reshape(shape)
    hh = h.reshape(shape)
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * y[i] + h10 * hh * m[i] + h01 * y[i + 1] + h11 * hh * m[i + 1]


def keyframe_matrix(
    keyframes: Sequence[Keyframe], joint_names: Sequence[str], home_pose_rad: np.ndarray
) -> np.ndarray:
    """``(n_keyframes, n_joints)`` joint angles with home-pose defaults."""
    index = {n: i for i, n in enumerate(joint_names)}
    out = np.tile(np.asarray(home_pose_rad, dtype=np.float64), (len(keyframes), 1))
    for r, k in enumerate(keyframes):
        for name, value in k.joints_rad.items():
            out[r, index[name]] = value
    return out


def interpolate_keyframes(
    keyframes: Sequence[Keyframe],
    joint_names: Sequence[str],
    home_pose_rad: np.ndarray,
    rate_hz: float,
) -> KeyframeTrajectory:
    """Sample the keyframe motion on a uniform grid at ``rate_hz`` (inclusive of the end)."""
    knots = np.array([k.time_s for k in keyframes])
    n = int(round(knots[-1] * rate_hz)) + 1
    times = np.arange(n) / rate_hz
    joints = pchip_eval(knots, keyframe_matrix(keyframes, joint_names, home_pose_rad), times)
    pitch = pchip_eval(knots, np.array([k.pitch_rad for k in keyframes]), times)
    return KeyframeTrajectory(times, joints, pitch, contact_at(keyframes, times))


def contact_at(keyframes: Sequence[Keyframe], times_s: np.ndarray) -> np.ndarray:
    """Contact flag at each time; a segment is in contact iff both ends are."""
    knots = np.array([k.time_s for k in keyframes])
    flags = np.array([k.in_contact for k in keyframes])
    i = np.clip(np.searchsorted(knots, times_s, side="right") - 1, 0, len(knots) - 1)
    at_knot = np.isclose(times_s, knots[i])
    nxt = np.minimum(i + 1, len(knots) - 1)
    return np.where(at_knot, flags[i], flags[i] & flags[nxt])


def flight_phase(keyframes: Sequence[Keyframe]) -> FlightPhase:
    """The single ballistic phase: last contact keyframe before and first after it.

    Raises ``ValueError`` unless the motion has exactly one contiguous run of
    airborne keyframes.
    """
    flags = [k.in_contact for k in keyframes]
    airborne = [i for i, f in enumerate(flags) if not f]
    if not airborne:
        raise ValueError("motion has no airborne keyframes")
    first, last = airborne[0], airborne[-1]
    if last - first + 1 != len(airborne):
        raise ValueError("motion has more than one flight phase")
    if first == 0 or last == len(keyframes) - 1:
        raise ValueError("flight phase must be preceded and followed by contact")
    return FlightPhase(keyframes[first - 1].time_s, keyframes[last + 1].time_s)


def ballistic_takeoff_velocity(
    z0_m: float, z1_m: float, flight_time_s: float, gravity_mps2: float
) -> float:
    """Vertical take-off speed (m/s) so a point rises from ``z0`` and lands at ``z1``."""
    if flight_time_s <= 0.0:
        raise ValueError("flight time must be positive")
    return (z1_m - z0_m) / flight_time_s + 0.5 * gravity_mps2 * flight_time_s
