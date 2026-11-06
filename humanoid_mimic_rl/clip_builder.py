"""Turn a keyframed joint/pitch trajectory into a full reference clip (numpy only).

Keyframes specify joint angles, root pitch and foot contact. The root
translation is not authored; it is derived so the clip is kinematically
plausible:

* **stance** (feet in contact): the root height is chosen so the lowest foot
  contact point touches the floor (z = 0), and the root x so the mean foot x
  stays where the stance began (no foot sliding);
* **flight**: the whole-body centre of mass follows the ballistic parabola
  that connects the CoM at take-off to the CoM at touch-down under gravity,
  and the root is placed so the CoM lands on it. By default the landing foot
  position is chosen so the flight's horizontal CoM velocity equals the
  stance CoM velocity at take-off (horizontally consistent); passing
  ``landing_offset_x_m`` fixes the landing foot x relative to take-off instead.

Forward kinematics is injected through the ``Kinematics`` protocol so this
module stays testable without MuJoCo; ``mj_kinematics.MjKinematics`` is the
MuJoCo implementation used by ``scripts/make_reference.py``.

Assumptions (documented, not physically enforced): yaw and roll are zero,
the motion is in the x-z plane, and the velocity at take-off/touch-down is
not forced to be continuous (the reported ``takeoff_velocity_jump_mps``
quantifies the mismatch). Physical feasibility is left to the RL policy, as
in DeepMimic.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np

from humanoid_mimic_rl import keyframes as kf
from humanoid_mimic_rl import quat, reference

GRAVITY_MPS2 = 9.81


@dataclasses.dataclass(frozen=True)
class FKResult:
    """Forward-kinematics quantities in the world frame (m)."""

    com: np.ndarray  # (3,) whole-body centre of mass
    foot_points: np.ndarray  # (n, 3) lowest points of the foot contact geoms
    site_pos: np.ndarray  # (n_sites, 3) tracked end-effector sites


class Kinematics(Protocol):
    """Forward kinematics of a floating-base robot."""

    site_names: tuple[str, ...]

    def forward(self, qpos: np.ndarray) -> FKResult:
        """FK for generalized position ``qpos`` = [root pos, root quat wxyz, joints]."""
        ...


@dataclasses.dataclass(frozen=True)
class ClipDiagnostics:
    """Sanity numbers printed by ``make_reference.py``."""

    takeoff_vz_mps: float
    landing_offset_x_m: float
    flight_time_s: float
    apex_com_height_m: float
    takeoff_velocity_jump_mps: float
    landing_velocity_jump_mps: float
    # Lowest foot point over the whole clip; < 0 means the feet would
    # penetrate the floor during flight (keyframes need fixing).
    min_foot_height_m: float


def _root_quat_from_pitch(pitch_rad: np.ndarray) -> np.ndarray:
    axis = np.array([0.0, 1.0, 0.0])
    return quat.quat_from_axis_angle(axis, pitch_rad, np)


def build_clip(
    traj: kf.KeyframeTrajectory,
    flight: kf.FlightPhase,
    kin: Kinematics,
    joint_names: tuple[str, ...],
    landing_offset_x_m: float | None,
    licence: str,
    source: str,
) -> tuple[ClipDiagnostics, reference.ReferenceClip]:
    """Fit the root translation and return ``(diagnostics, clip)``."""
    times = traj.times_s
    n = len(times)
    rate_hz = 1.0 / float(times[1] - times[0])
    quats = _root_quat_from_pitch(traj.pitch_rad)

    # FK with the root at the origin: offsets of CoM / feet relative to the root.
    com_off = np.empty((n, 3))
    foot_min_z = np.empty(n)
    foot_mean_x = np.empty(n)
    for k in range(n):
        q = np.concatenate([np.zeros(3), quats[k], traj.joint_pos_rad[k]])
        fk = kin.forward(q)
        com_off[k] = fk.com
        foot_min_z[k] = np.min(fk.foot_points[:, 2])
        foot_mean_x[k] = np.mean(fk.foot_points[:, 0])

    root = np.zeros((n, 3))
    in_flight = (times > flight.takeoff_s + 1e-9) & (times < flight.landing_s - 1e-9)
    before = times <= flight.takeoff_s + 1e-9
    after = times >= flight.landing_s - 1e-9

    # Stance: lowest foot point on the floor, feet fixed in x.
    root[:, 2] = -foot_min_z
    stance_x_before = 0.0
    root[before, 0] = stance_x_before - foot_mean_x[before]
    i_to = int(np.flatnonzero(before)[-1])
    i_ld = int(np.flatnonzero(after)[0])
    t_flight = float(times[i_ld] - times[i_to])
    com_to = root[i_to] + com_off[i_to]
    # Landing CoM x for a zero landing offset, then shift the landing stance.
    com_ld_x0 = stance_x_before - foot_mean_x[i_ld] + com_off[i_ld, 0]
    if landing_offset_x_m is None:
        stance_vx = (com_to[0] - (root[i_to - 1, 0] + com_off[i_to - 1, 0])) * rate_hz
        landing_offset_x_m = com_to[0] + stance_vx * t_flight - com_ld_x0
    stance_x_after = stance_x_before + landing_offset_x_m
    root[after, 0] = stance_x_after - foot_mean_x[after]
    com_ld = root[i_ld] + com_off[i_ld]

    # Flight: CoM on the ballistic parabola between take-off and touch-down.
    vz0 = kf.ballistic_takeoff_velocity(com_to[2], com_ld[2], t_flight, GRAVITY_MPS2)
    vx = (com_ld[0] - com_to[0]) / t_flight
    tau = times[in_flight] - times[i_to]
    com_flight = np.stack(
        [
            com_to[0] + vx * tau,
            np.full_like(tau, com_to[1]),
            com_to[2] + vz0 * tau - 0.5 * GRAVITY_MPS2 * tau**2,
        ],
        axis=-1,
    )
    root[in_flight] = com_flight - com_off[in_flight]

    qpos = np.concatenate([root, quats, traj.joint_pos_rad], axis=1)
    qvel = reference.finite_difference_qvel(qpos, rate_hz)

    site_pos = np.stack([kin.forward(qpos[k]).site_pos for k in range(n)])
    min_foot = float(np.min(root[:, 2] + foot_min_z))

    com = root + com_off
    ballistic_v_to = np.array([vx, 0.0, vz0])
    ballistic_v_ld = np.array([vx, 0.0, vz0 - GRAVITY_MPS2 * t_flight])
    # Stance-side CoM velocity from one-sided differences across the boundary.
    stance_v_to = (com[i_to] - com[i_to - 1]) * rate_hz
    stance_v_ld = (com[i_ld + 1] - com[i_ld]) * rate_hz
    diagnostics = ClipDiagnostics(
        takeoff_vz_mps=float(vz0),
        landing_offset_x_m=float(landing_offset_x_m),
        flight_time_s=t_flight,
        apex_com_height_m=float(np.max(com[:, 2])),
        takeoff_velocity_jump_mps=float(np.linalg.norm(ballistic_v_to - stance_v_to)),
        landing_velocity_jump_mps=float(np.linalg.norm(ballistic_v_ld - stance_v_ld)),
        min_foot_height_m=min_foot,
    )

    clip = reference.ReferenceClip(
        fps_hz=rate_hz,
        qpos=qpos,
        qvel=qvel,
        joint_names=tuple(joint_names),
        site_names=tuple(kin.site_names),
        site_pos=site_pos,
        licence=licence,
        source=source,
    )
    clip.validate()
    return diagnostics, clip
