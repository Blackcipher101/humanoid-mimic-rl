"""Torque / contact feasibility of a G1 clip with feet **and hands** as contacts (mujoco + scipy, no JAX).

``flip_dynamics.check_clip`` splits the root wrench equally between the two feet, which is only valid for
left/right-symmetric motions and has no hand contact, so it cannot judge cartwheels. This module reuses the
same ``FloatingBaseModel`` (constraint-free ``mj_inverse``: ``qfrc_inverse = M qacc + c - passive``, the
generalized force that actuators plus contacts supply; same finite-difference ``qvel`` / ``qacc``; same
transition-frame exclusion; same ``actuator_caps`` = the published peak torques at torque scale 1) and
replaces the equal split by a per-frame linear programme over the real contact set:

* contact points: 4 sole corners per foot touching the floor (within ``CONTACT_TOL_M``) and the 2 capsule end
  points (lowest surface points) of each hand touching it; each carries a force in an inscribed
  4-sided friction pyramid (``mu`` = ``g1_model.FOOT_FRICTION``, normal force >= 0), so a feasible
  distribution automatically has the centre of pressure inside the support polygon;
* stage A: minimise the L1 residual between the root wrench ``qfrc_inverse[:6]`` and what the contacts can
  supply. With no contact this is the flight root wrench itself (must be ~0 for a ballistic flight);
* stage B: with that residual fixed, minimise the largest ``|tau_j| / cap_j`` over the joints, where
  ``tau = qfrc_inverse[6:] - J_c^T lambda``. The torque ratio reported is therefore the best case over all
  admissible contact force distributions (a lower bound on what any controller would need).

Frames: world z-up, SI units; ``qfrc[0:3]`` world force, ``qfrc[3:6]`` body-frame torque about the root, as in
``flip_dynamics``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import mujoco
import numpy as np
from scipy.optimize import linprog

from humanoid_mimic_rl import flip_dynamics as fd
from humanoid_mimic_rl import g1_model, reference

CONTACT_TOL_M = 0.01  # a contact point this close to the floor can carry load
HAND_RADIUS_M = 0.05
EE_NAMES: tuple[str, ...] = ("lfoot", "rfoot", "lhand", "rhand")
ROOT_FORCE_TOL_FRAC = 0.02  # same criteria as scripts/make_g1_backflip.py
ROOT_TORQUE_TOL_NM = 2.0
TORQUE_MARGIN = 0.85


@dataclasses.dataclass(frozen=True)
class ContactGeometry:
    """Where the G1 can touch the floor: sole corners of the feet and capsule ends of the hands."""

    foot_corner_local: np.ndarray  # (4, 3) in the ankle-roll frame
    foot_bodies: tuple[int, int]
    hand_geoms: tuple[int, int]
    hand_half_length_m: float


def contact_geometry(fb: fd.FloatingBaseModel) -> ContactGeometry:
    m = fb.m
    sx, sy, sz = g1_model.FOOT_SOLE_CENTRE_LOCAL_M
    hx, hy = g1_model.FOOT_HALF_LENGTH_M, g1_model.FOOT_HALF_WIDTH_M
    corners = np.array([[sx + a * hx, sy + b * hy, sz] for a in (-1, 1) for b in (-1, 1)])
    hands = (int(m.geom("left_hand_collision").id), int(m.geom("right_hand_collision").id))
    return ContactGeometry(
        corners, (int(m.body(g1_model.FOOT_BODIES[0]).id), int(m.body(g1_model.FOOT_BODIES[1]).id)), hands,
        float(m.geom_size[hands[0], 1]),
    )  # fmt: skip


def _points(fb: fd.FloatingBaseModel, geo: ContactGeometry, ee: int) -> tuple[np.ndarray, int]:
    """World contact points ``(n, 3)`` of end effector ``ee`` (0..3) and the body they belong to; needs kinematics."""
    d = fb.d
    if ee < 2:
        b = geo.foot_bodies[ee]
        return d.xpos[b] + geo.foot_corner_local @ d.xmat[b].reshape(3, 3).T, b
    g = geo.hand_geoms[ee - 2]
    rot = d.geom_xmat[g].reshape(3, 3)
    ends = np.array([[0.0, 0.0, geo.hand_half_length_m], [0.0, 0.0, -geo.hand_half_length_m]])
    pts = d.geom_xpos[g] + ends @ rot.T
    pts[:, 2] -= HAND_RADIUS_M  # lowest surface point of the capsule end sphere
    return pts, int(fb.m.geom_bodyid[g])


def contact_flags(fb: fd.FloatingBaseModel, geo: ContactGeometry, qpos: np.ndarray) -> np.ndarray:
    """``(T, 4)`` bool: end effector ``lfoot rfoot lhand rhand`` has a point within ``CONTACT_TOL_M`` of the floor."""
    out = np.zeros((qpos.shape[0], 4), dtype=bool)
    for k in range(qpos.shape[0]):
        fb.kinematics(qpos[k])
        for ee in range(4):
            out[k, ee] = _points(fb, geo, ee)[0][:, 2].min() <= CONTACT_TOL_M
    return out


def _friction_generators(mu: float) -> np.ndarray:
    """Inscribed friction pyramid edges ``(4, 3)``: ``n + mu * (+-x, +-y)``."""
    return np.array([[mu, 0.0, 1.0], [-mu, 0.0, 1.0], [0.0, mu, 1.0], [0.0, -mu, 1.0]])


def contact_basis(
    fb: fd.FloatingBaseModel, geo: ContactGeometry, active: Sequence[int], mu: float
) -> np.ndarray:
    """Generalized forces ``(nv, 4 n_points)`` of unit friction-pyramid edge forces at the active contact points.

    Call after ``fb.inverse`` (kinematics and ``mj_comPos`` are then current)."""
    gens = _friction_generators(mu)
    cols = []
    jac = np.zeros((3, fb.nv))
    for ee in active:
        pts, body = _points(fb, geo, ee)
        for p in pts:
            jac[:] = 0.0
            mujoco.mj_jac(fb.m, fb.d, jac, None, p, body)
            cols.append(jac.T @ gens.T)  # (nv, 4)
    return np.concatenate(cols, axis=1) if cols else np.zeros((fb.nv, 0))


def solve_frame(
    f: np.ndarray, basis: np.ndarray, cap_pos: np.ndarray, cap_neg: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """Two-stage LP for one frame. Returns ``(root wrench residual (6,), max torque ratio, per-joint ratio)``.

    ``f`` = ``qfrc_inverse`` (nv,), ``basis`` = contact generalized forces (nv, n)."""
    n = basis.shape[1]
    a_root = basis[:6]
    # Stage A: min |r|_1 s.t. a_root beta + r+ - r- = f_root.
    c = np.concatenate([np.zeros(n), np.ones(12)])
    a_eq = np.hstack([a_root, np.eye(6), -np.eye(6)])
    res = linprog(c, A_eq=a_eq, b_eq=f[:6], bounds=[(0, None)] * (n + 12), method="highs")
    if res.status != 0:
        raise RuntimeError(f"contact LP stage A failed: {res.message}")
    resid = res.x[n : n + 6] - res.x[n + 6 :]
    # Stage B: min t s.t. a_root beta = f_root - resid, -t cap_neg <= f_j - B_j beta <= t cap_pos.
    b_j = basis[6:]
    tau0 = f[6:]
    c = np.concatenate([np.zeros(n), [1.0]])
    a_ub = np.vstack([np.hstack([-b_j, -cap_pos[:, None]]), np.hstack([b_j, -cap_neg[:, None]])])
    b_ub = np.concatenate([-tau0, tau0])
    a_eq = np.hstack([a_root, np.zeros((6, 1))])
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=f[:6] - resid, bounds=[(0, None)] * (n + 1), method="highs")
    if res.status != 0:
        raise RuntimeError(f"contact LP stage B failed: {res.message}")
    tau = tau0 - b_j @ res.x[:n]
    ratio = np.abs(tau) / np.where(tau >= 0.0, cap_pos, cap_neg)
    return resid, float(res.x[n]), ratio


@dataclasses.dataclass(frozen=True)
class FeasibilityReport:
    frames: int
    excluded_frames: int
    flight_frames: int
    contact_frames: int
    contact_frames_by_end_effector: dict[str, int]
    body_weight_n: float
    flight_root_force_max_n: float
    flight_root_torque_max_nm: float
    contact_wrench_residual_max_n: float  # wrench the contacts cannot supply (unilateral / friction / support)
    contact_wrench_residual_max_nm: float
    torque_ratio_max: float
    torque_ratio_by_joint: dict[str, float]
    worst_joint: str
    worst_frame: int
    joints_over_limit: list[str]

    def passes(self, torque_margin: float, root_force_tol_n: float, root_torque_tol_nm: float) -> bool:
        """Flight root wrench ~0, contact wrench realisable, torques within ``torque_margin`` of the peaks."""
        return (
            self.flight_root_force_max_n <= root_force_tol_n
            and self.flight_root_torque_max_nm <= root_torque_tol_nm
            and self.contact_wrench_residual_max_n <= root_force_tol_n
            and self.contact_wrench_residual_max_nm <= root_torque_tol_nm
            and self.torque_ratio_max <= torque_margin
        )


def check_clip(
    fb: fd.FloatingBaseModel,
    qpos: np.ndarray,
    fps_hz: float,
    joint_names: Sequence[str] = g1_model.G1_JOINT_NAMES,
    mu: float = g1_model.FOOT_FRICTION,
) -> FeasibilityReport:
    """Inverse-dynamics feasibility of ``qpos`` ``(T, 7 + 29)`` at ``fps_hz`` with feet and hands as contacts."""
    geo = contact_geometry(fb)
    flags = contact_flags(fb, geo, qpos)
    any_contact = flags.any(axis=1)
    qvel = reference.finite_difference_qvel(qpos, fps_hz)
    qacc = np.gradient(qvel, 1.0 / fps_hz, axis=0)
    n = qpos.shape[0]
    changed = np.zeros(n, dtype=bool)
    for i in np.flatnonzero((flags[1:] != flags[:-1]).any(axis=1)):
        changed[max(i - 1, 0) : min(i + 3, n)] = True
    changed[[0, 1, -2, -1]] = True  # one-sided differences at the clip ends
    flight_f, flight_t, res_f, res_t = [0.0], [0.0], [0.0], [0.0]
    ratio = np.zeros(fb.m.nu)
    worst_ratio, worst_frame = 0.0, 0
    for k in range(n):
        if changed[k]:
            continue
        f = fb.inverse(qpos[k], qvel[k], qacc[k])
        cap_pos, cap_neg = fb.actuator_caps(qpos[k, 7:])
        active = [ee for ee in range(4) if flags[k, ee]]
        basis = contact_basis(fb, geo, active, mu)
        resid, _, r_k = solve_frame(f, basis, cap_pos, cap_neg)
        if active:
            res_f.append(float(np.linalg.norm(resid[:3])))
            res_t.append(float(np.linalg.norm(resid[3:])))
        else:
            flight_f.append(float(np.linalg.norm(f[:3])))
            flight_t.append(float(np.linalg.norm(f[3:6])))
        ratio = np.maximum(ratio, r_k)
        if r_k.max() > worst_ratio:
            worst_ratio, worst_frame = float(r_k.max()), k
    by_joint = {name: float(r) for name, r in zip(joint_names, ratio)}
    worst = max(by_joint, key=by_joint.get)
    return FeasibilityReport(
        frames=n,
        excluded_frames=int(changed.sum()),
        flight_frames=int(np.sum(~any_contact & ~changed)),
        contact_frames=int(np.sum(any_contact & ~changed)),
        contact_frames_by_end_effector={ee: int(np.sum(flags[~changed, i])) for i, ee in enumerate(EE_NAMES)},
        body_weight_n=fb.mass_kg * fd.GRAVITY_MPS2,
        flight_root_force_max_n=float(np.max(flight_f)),
        flight_root_torque_max_nm=float(np.max(flight_t)),
        contact_wrench_residual_max_n=float(np.max(res_f)),
        contact_wrench_residual_max_nm=float(np.max(res_t)),
        torque_ratio_max=float(np.max(ratio)),
        torque_ratio_by_joint=by_joint,
        worst_joint=worst,
        worst_frame=worst_frame,
        joints_over_limit=[j for j, r in by_joint.items() if r > 1.0],
    )


def com_position(fb: fd.FloatingBaseModel, qpos: np.ndarray) -> np.ndarray:
    """World centre of mass (3,) of the whole robot at ``qpos``."""
    fb.kinematics(qpos)
    mujoco.mj_comPos(fb.m, fb.d)
    return fb.d.subtree_com[1].copy()


def ballistic_flight_fix(fb: fd.FloatingBaseModel, qpos: np.ndarray, fps_hz: float) -> tuple[np.ndarray, int]:
    """Make every flight segment's centre of mass exactly ballistic (``a = -g``).

    A flight segment is a maximal run of frames with no end effector touching the floor. The CoM of the frames
    just before and after it is kept; in between the CoM follows the parabola through those two states and the
    root translation is shifted by the difference (joint angles and root orientation untouched). The
    CoM velocity jumps at the two boundary frames, which ``check_clip`` excludes like any contact change.
    Returns ``(new qpos, number of flight segments)``.
    """
    flags = contact_flags(fb, contact_geometry(fb), qpos).any(axis=1)
    out = qpos.copy()
    com = np.array([com_position(fb, q) for q in qpos])
    n_seg = 0
    k = 0
    while k < len(flags):
        if flags[k]:
            k += 1
            continue
        a = k
        while k < len(flags) and not flags[k]:
            k += 1
        b = k  # first frame after the segment (or len)
        if a == 0 or b >= len(flags):
            continue  # not bounded by contacts on both sides
        n_seg += 1
        t = (np.arange(a - 1, b + 1) - (a - 1)) / fps_hz
        dur = t[-1]
        g = np.array([0.0, 0.0, -fd.GRAVITY_MPS2])
        v0 = (com[b] - com[a - 1] - 0.5 * g * dur**2) / dur
        ballistic = com[a - 1] + np.outer(t, v0) + 0.5 * np.outer(t**2, g)
        out[a - 1 : b + 1, :3] += ballistic - com[a - 1 : b + 1]
    return out, n_seg
