"""Dynamically consistent standing-backflip reference by trajectory optimisation (mujoco + scipy).

Decision variables: knot values of a clamped C2 cubic spline per *joint group*
(left/right-symmetric sagittal joints, e.g. hip pitch, knee, ankle pitch,
waist pitch, shoulder pitch, elbow; every other joint stays at the home pose)
and the take-off time.

The floating base is never a decision variable; it follows from physics:

* **stance** (both feet flat and fixed on the floor): the root pose is the one
  that keeps the left foot at its anchor (the motion is left/right symmetric,
  so the right foot follows). Root velocity/acceleration are central time
  differences of that pose;
* **flight**: the root is integrated (RK4) from the take-off state with **zero
  external wrench** under the prescribed joint motion, so the CoM is ballistic
  and the whole-body angular momentum about the CoM is conserved by
  construction;
* **touch-down** is the event at which the foot origin returns to its standing
  height. Constraints require the foot to be flat there (pitch exactly -2*pi)
  and slow (bounded foot velocity, i.e. a small impact); the landing stance is
  anchored where the foot touched down.

Inverse dynamics (``flip_dynamics``) gives the constraints: joint torques
within ``torque_margin`` x actuator caps (force limit and PD authority),
positive normal force, friction cone, centre of pressure on the sole, flight
clearance of feet and body, joint limits. The effort objective is the mean
squared torque/cap ratio.

``solve_staged`` runs bounded least squares on constraint violations, stage by
stage (push-off only, then everything), with forward-difference Jacobians
evaluated in a process pool.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from humanoid_mimic_rl import flip_dynamics as fd
from humanoid_mimic_rl import quat, reference

TWO_PI = 2.0 * np.pi


@dataclasses.dataclass(frozen=True)
class FlipSetup:
    """Static problem description (SI units)."""

    groups: dict[str, tuple[int, ...]]  # group -> indices into the joint vector (qpos[7:])
    home_qpos: np.ndarray  # (nq,) standing pose with feet flat on the floor
    knot_times_s: np.ndarray  # (K,): first two and last two knots pinned to the home pose
    duration_s: float
    sample_rate_hz: float = 100.0
    takeoff_bounds_s: tuple[float, float] = (0.8, 1.2)
    min_flight_s: float = 0.25
    max_flight_s: float = 0.9
    takeoff_vz_target_mps: float = 2.3  # stage-A target for the CoM vertical speed
    # Stage-A target for the angular momentum about the CoM, world y (negative =
    # backwards). ~ rotation / integral(1 / I_yy dt) with I_yy ~ 1-5 kg m^2.
    takeoff_ang_mom_target_nms: float = -17.0
    takeoff_vx_target_mps: float = -0.5  # stage-A target: modest backward travel
    torque_margin: float = 0.85
    # Cap torques by the position servo's PD authority (kp * distance to the
    # control limit) as well as the force limit. True matches what the
    # simulated G1 can actually produce; False uses force limits only.
    use_pd_authority: bool = True
    friction: float = 0.6
    friction_margin: float = 0.8
    cop_margin: float = 0.8
    min_normal_force_frac: float = 0.05  # of body weight, per foot
    # Touch-down is allowed to be a small impact (the stance after it starts
    # from the flight state; the residual velocity jump is reported).
    max_touchdown_foot_speed_mps: tuple[float, float] = (0.3, 1.0)  # |vx|, |vz|
    max_touchdown_foot_pitch_rate_radps: float = 2.0
    foot_clearance_m: float = 0.01
    body_clearance_m: float = 0.02
    event_guard_s: float = 0.05  # no clearance check this close to take-off / touch-down
    joint_limit_margin_rad: float = 0.02
    rk4_step_s: float = 0.0025
    fd_eps_s: float = 1e-3

    @property
    def home_joint_pos(self) -> np.ndarray:
        return self.home_qpos[7:]

    @property
    def free_knots(self) -> np.ndarray:
        return np.arange(2, len(self.knot_times_s) - 2)

    @property
    def num_params(self) -> int:
        return len(self.free_knots) * len(self.groups) + 1


@dataclasses.dataclass
class Rollout:
    """Trajectory sampled at ``setup.sample_rate_hz`` plus its inverse dynamics."""

    times_s: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    contact: np.ndarray
    takeoff_s: float
    landing_s: float
    landed: bool
    tau: np.ndarray  # (T, nu) actuator torques
    torque_ratio: np.ndarray  # (T, nu) |tau| / cap (direction-dependent cap)
    foot_wrench: np.ndarray  # (T, 6) per-foot wrench in stance, NaN in flight
    clearance: np.ndarray  # (T,) min clearance minus margin in flight, +inf elsewhere
    takeoff_com_vel: np.ndarray  # (3,)
    takeoff_ang_mom: np.ndarray  # (3,) about the CoM, world frame
    touchdown_foot_pitch: float  # unwrapped (rad)
    touchdown_foot_vel: np.ndarray  # (3,) vx, vz (m/s), pitch rate (rad/s)


class FlipProblem:
    def __init__(self, fb: fd.FloatingBaseModel, setup: FlipSetup) -> None:
        self.fb = fb
        self.s = setup
        fb.kinematics(setup.home_qpos)
        b = fb.foot_ids[0]
        self.anchor0 = fb.d.xpos[b].copy()
        if abs(fd.pitch_of(fb.d.xmat[b].reshape(3, 3))) > 1e-3:
            raise ValueError("home pose must have flat feet")
        self.n_group = len(setup.groups)

    # ------------------------------------------------------------------
    # Parameterisation.
    # ------------------------------------------------------------------

    def group_home(self) -> np.ndarray:
        return np.array([self.s.home_joint_pos[idx[0]] for idx in self.s.groups.values()])

    def knot_matrix(self, x: np.ndarray) -> np.ndarray:
        s = self.s
        k = np.tile(self.group_home(), (len(s.knot_times_s), 1))
        k[s.free_knots] = x[:-1].reshape(len(s.free_knots), self.n_group)
        return k

    def pack(self, free_knot_values: np.ndarray, takeoff_s: float) -> np.ndarray:
        return np.concatenate([np.asarray(free_knot_values, dtype=np.float64).reshape(-1), [takeoff_s]])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        fb, lo, hi = self.fb, [], []
        for _ in self.s.free_knots:
            for idx in self.s.groups.values():
                lo.append(fb.jnt_lo[idx[0]] + 0.05)
                hi.append(fb.jnt_hi[idx[0]] - 0.05)
        return np.array(lo + [self.s.takeoff_bounds_s[0]]), np.array(hi + [self.s.takeoff_bounds_s[1]])

    def knot_param_index(self, knot: int, group: str) -> int:
        """Index in ``x`` of the value of ``group`` at knot ``knot`` (must be free)."""
        k = int(np.flatnonzero(self.s.free_knots == knot)[0])
        return k * self.n_group + list(self.s.groups).index(group)

    def joint_spline(self, x: np.ndarray) -> CubicSpline:
        return CubicSpline(self.s.knot_times_s, self.knot_matrix(x), bc_type="clamped", axis=0)

    def _full(self, red: np.ndarray, base: np.ndarray) -> np.ndarray:
        out = base.copy()
        for g, idx in enumerate(self.s.groups.values()):
            out[list(idx)] = red[g]
        return out

    def joints_at(self, spl: CubicSpline, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zero = np.zeros_like(self.s.home_joint_pos)
        return (
            self._full(spl(t), self.s.home_joint_pos),
            self._full(spl(t, 1), zero),
            self._full(spl(t, 2), zero),
        )

    # ------------------------------------------------------------------
    # Root motion.
    # ------------------------------------------------------------------

    def stance_root(self, q_joint: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Root position and quaternion with the left foot flat at ``anchor`` (yaw 0)."""
        q = np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0], q_joint])
        p_f, r_f = self.fb.foot_pose(q, 0)
        r_b = r_f.T
        return anchor - r_b @ p_f, fd.mat_to_quat(r_b)

    def stance_state(self, spl: CubicSpline, t: float, anchor: np.ndarray):
        e = self.s.fd_eps_s
        qj, qdj, qddj = self.joints_at(spl, t)
        p0, q0 = self.stance_root(qj, anchor)
        pm, qm = self.stance_root(self.joints_at(spl, t - e)[0], anchor)
        pp, qp = self.stance_root(self.joints_at(spl, t + e)[0], anchor)
        qm = qm if np.dot(qm, q0) >= 0 else -qm
        qp = qp if np.dot(qp, q0) >= 0 else -qp
        w, alpha = fd.body_rates(qm, q0, qp, e)
        qpos = np.concatenate([p0, q0, qj])
        qvel = np.concatenate([(pp - pm) / (2 * e), w, qdj])
        qacc = np.concatenate([(pp - 2 * p0 + pm) / e**2, alpha, qddj])
        return qpos, qvel, qacc

    def _deriv(self, spl: CubicSpline, t: float, y: np.ndarray) -> np.ndarray:
        qj, qdj, qddj = self.joints_at(spl, t)
        p, qr, v, w = y[0:3], y[3:7], y[7:10], y[10:13]
        a = self.fb.root_accel(np.concatenate([p, qr, qj]), np.concatenate([v, w, qdj]), qddj)
        qdot = 0.5 * quat.quat_mul(qr, np.concatenate([[0.0], w]), np)
        return np.concatenate([v, qdot, a[0:3], a[3:6]])

    def _rk4(self, spl: CubicSpline, t: float, y: np.ndarray, h: float) -> np.ndarray:
        k1 = self._deriv(spl, t, y)
        k2 = self._deriv(spl, t + h / 2, y + h / 2 * k1)
        k3 = self._deriv(spl, t + h / 2, y + h / 2 * k2)
        k4 = self._deriv(spl, t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y[3:7] /= np.linalg.norm(y[3:7])
        return y

    def _foot_height(self, spl: CubicSpline, t: float, y: np.ndarray) -> float:
        qj = self.joints_at(spl, t)[0]
        self.fb.kinematics(np.concatenate([y[0:3], y[3:7], qj]))
        return float(self.fb.d.xpos[self.fb.foot_ids[0], 2] - self.anchor0[2])

    def flight(self, spl: CubicSpline, t0: float, y0: np.ndarray):
        """Integrate from take-off until the foot returns to standing height (or ``max_flight_s``).

        Returns ``(times, states)`` on the RK4 grid including the exact touch-down
        state, and whether touch-down happened.
        """
        s = self.s
        ts, ys = [t0], [y0.copy()]
        t, y = t0, y0.copy()
        h_prev = self._foot_height(spl, t, y)
        while t < t0 + s.max_flight_s - 1e-12:
            y_new = self._rk4(spl, t, y, s.rk4_step_s)
            t_new = t + s.rk4_step_s
            h_new = self._foot_height(spl, t_new, y_new)
            if t_new - t0 >= s.min_flight_s and h_prev > 0.0 >= h_new:
                lo, hi, y_lo = 0.0, s.rk4_step_s, y  # bisection on the step length
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    if self._foot_height(spl, t + mid, self._rk4(spl, t, y_lo, mid)) > 0.0:
                        lo = mid
                    else:
                        hi = mid
                ts.append(t + hi)
                ys.append(self._rk4(spl, t, y_lo, hi))
                return np.array(ts), np.array(ys), True
            t, y, h_prev = t_new, y_new, h_new
            ts.append(t)
            ys.append(y.copy())
        return np.array(ts), np.array(ys), False

    # ------------------------------------------------------------------
    # Full rollout.
    # ------------------------------------------------------------------

    def rollout(self, x: np.ndarray, takeoff_only: bool = False) -> Rollout:
        s, fb = self.s, self.fb
        spl = self.joint_spline(x)
        t_to = float(x[-1])
        qp0, qv0, _ = self.stance_state(spl, t_to, self.anchor0)
        com_vel, ang_mom = self._momentum(qp0, qv0)
        end = t_to if takeoff_only else s.duration_s
        times = np.arange(int(np.floor(end * s.sample_rate_hz + 1e-9)) + 1) / s.sample_rate_hz

        y0 = np.concatenate([qp0[0:3], qp0[3:7], qv0[0:3], qv0[3:6]])
        if takeoff_only:
            ft, fy, landed, t_ld = np.zeros(0), np.zeros((0, 13)), False, np.inf
        else:
            ft, fy, landed = self.flight(spl, t_to, y0)
            t_ld = float(ft[-1])
        contact = (times <= t_to) | (times >= t_ld)
        n, nq, nv = len(times), fb.m.nq, fb.m.nv
        qpos, qvel, qacc = np.zeros((n, nq)), np.zeros((n, nv)), np.zeros((n, nv))
        for k in np.flatnonzero(times <= t_to):
            qpos[k], qvel[k], qacc[k] = self.stance_state(spl, times[k], self.anchor0)

        td_pitch, td_vel = np.nan, np.full(3, np.nan)
        if not takeoff_only:
            for k in np.flatnonzero(~contact):  # flight samples: interpolate the RK4 grid by re-integration
                i = int(np.searchsorted(ft, times[k], side="right") - 1)
                y = self._rk4(spl, ft[i], fy[i], times[k] - ft[i]) if times[k] > ft[i] else fy[i]
                qj, qdj, qddj = self.joints_at(spl, times[k])
                qpos[k] = np.concatenate([y[0:3], y[3:7], qj])
                qvel[k] = np.concatenate([y[7:10], y[10:13], qdj])
                qacc[k] = np.concatenate([fb.root_accel(qpos[k], qvel[k], qddj), qddj])
            y_ld = fy[-1]
            qj, qdj, _ = self.joints_at(spl, t_ld)
            qpos_ld = np.concatenate([y_ld[0:3], y_ld[3:7], qj])
            vel6 = self._foot_velocity(qpos_ld, np.concatenate([y_ld[7:10], y_ld[10:13], qdj]))
            td_vel = np.array([vel6[3], vel6[5], vel6[1]])
            td_pitch = self._unwrapped_foot_pitch(fy, qpos_ld)
            fb.kinematics(qpos_ld)
            anchor1 = np.array([fb.d.xpos[fb.foot_ids[0], 0], self.anchor0[1], self.anchor0[2]])
            for k in np.flatnonzero(times >= t_ld):
                qpos[k], qvel[k], qacc[k] = self.stance_state(spl, times[k], anchor1)
        for k in range(1, n):  # keep the root quaternion sign continuous
            if np.dot(qpos[k, 3:7], qpos[k - 1, 3:7]) < 0:
                qpos[k, 3:7] *= -1

        nu = fb.m.nu
        tau, ratio = np.zeros((n, nu)), np.zeros((n, nu))
        wrench = np.full((n, 6), np.nan)
        clearance = np.full(n, np.inf)
        for k in range(n):
            f = fb.inverse(qpos[k], qvel[k], qacc[k])
            if contact[k]:
                wrench[k], tau[k] = fb.foot_wrench(f)
            else:
                tau[k] = f[6:]
                if t_to + s.event_guard_s < times[k] < t_ld - s.event_guard_s:
                    fb.kinematics(qpos[k])
                    feet = min(np.min(fb.sole_corner_heights(i)) for i in range(len(fb.foot_ids)))
                    clearance[k] = min(feet - s.foot_clearance_m, fb.body_clearance() - s.body_clearance_m)
            cap_pos, cap_neg = (
                fb.actuator_caps(qpos[k, 7:]) if s.use_pd_authority else (fb.force_limit, fb.force_limit)
            )
            # Caps floored at 1 N m: at a joint limit the PD authority is 0 and the ratio would explode.
            ratio[k] = np.abs(tau[k]) / np.maximum(np.where(tau[k] >= 0, cap_pos, cap_neg), 1.0)
        return Rollout(
            times, qpos, qvel, qacc, contact, t_to, t_ld, landed, tau, ratio, wrench, clearance,
            com_vel, ang_mom, float(td_pitch), td_vel,
        )

    def _momentum(self, qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Whole-body CoM velocity and angular momentum about the CoM (world frame)."""
        fb = self.fb
        fb.d.qpos[:] = qpos
        fb.d.qvel[:] = qvel
        mujoco.mj_fwdPosition(fb.m, fb.d)
        mujoco.mj_fwdVelocity(fb.m, fb.d)
        mujoco.mj_subtreeVel(fb.m, fb.d)
        return fb.d.subtree_linvel[1].copy(), fb.d.subtree_angmom[1].copy()

    def _foot_velocity(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        fb = self.fb
        fb.d.qpos[:] = qpos
        fb.d.qvel[:] = qvel
        mujoco.mj_fwdPosition(fb.m, fb.d)
        mujoco.mj_fwdVelocity(fb.m, fb.d)
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(fb.m, fb.d, mujoco.mjtObj.mjOBJ_BODY, int(fb.foot_ids[0]), vel6, 0)
        return vel6

    def _unwrapped_foot_pitch(self, flight_states: np.ndarray, qpos_ld: np.ndarray) -> float:
        """Foot pitch at touch-down, with the root rotation accumulated over the flight."""
        prev = flight_states[0, 3:7]
        angle = 2.0 * np.arctan2(prev[2], prev[0])
        for qr in flight_states[1:, 3:7]:
            angle += quat.quat_to_rotvec(quat.quat_mul(quat.quat_conj(prev, np), qr, np), np)[1]
            prev = qr
        self.fb.kinematics(np.concatenate([np.zeros(3), [1.0, 0, 0, 0], qpos_ld[7:]]))
        return angle + fd.pitch_of(self.fb.d.xmat[self.fb.foot_ids[0]].reshape(3, 3))

    # ------------------------------------------------------------------
    # Constraint evaluation.
    # ------------------------------------------------------------------

    def constraints(self, r: Rollout, torque_margin: float | None = None) -> dict[str, np.ndarray]:
        """Named constraint arrays, **per sample**; ``eq_*`` must be 0, ``ineq_*`` >= 0.

        All entries are scaled to be dimensionless (O(1) at the tolerance), so
        least squares sees every violating sample rather than only the worst.
        """
        s, fb = self.s, self.fb
        w = r.foot_wrench[r.contact] / (fb.mass_kg * fd.GRAVITY_MPS2)  # per-foot wrench, body weights
        fz = w[:, 2]
        mu = s.friction * s.friction_margin
        out = {
            "ineq_torque": ((s.torque_margin if torque_margin is None else torque_margin) - r.torque_ratio).ravel(),
            "ineq_normal": (fz - s.min_normal_force_frac / 2) * 10.0,
            "ineq_friction": (mu * fz - np.hypot(w[:, 0], w[:, 1])) * 10.0,
            "ineq_cop_x": (s.cop_margin * fz - np.abs(w[:, 4]) / fb.feet.half_length_m) * 10.0,
            "ineq_cop_y": (s.cop_margin * fz - np.abs(w[:, 3]) / fb.feet.half_width_m) * 10.0,
            "ineq_joint_lo": ((r.qpos[:, 7:] - fb.jnt_lo - s.joint_limit_margin_rad) * 10.0).ravel(),
            "ineq_joint_hi": ((fb.jnt_hi - r.qpos[:, 7:] - s.joint_limit_margin_rad) * 10.0).ravel(),
        }
        if np.isfinite(r.landing_s):  # full rollout (flight simulated)
            fin = r.clearance[np.isfinite(r.clearance)]
            vx, vz = s.max_touchdown_foot_speed_mps
            out["ineq_clearance"] = fin / 0.01
            out["ineq_touchdown"] = np.array(
                [
                    (vx - abs(r.touchdown_foot_vel[0])) / 0.05,
                    (vz - abs(r.touchdown_foot_vel[1])) / 0.05,
                    (s.max_touchdown_foot_pitch_rate_radps - abs(r.touchdown_foot_vel[2])) / 0.2,
                ]
            )
            out["eq_landing_pitch"] = np.array([(r.touchdown_foot_pitch + TWO_PI) / 0.05])
            out["ineq_landed"] = np.array([1.0 if r.landed else -10.0])
        return out

    def violations(self, r: Rollout) -> dict[str, float]:
        """Worst scaled violation per constraint group (0 = satisfied)."""
        return {
            k: float(np.max(np.abs(v)) if k.startswith("eq_") else max(0.0, -float(np.min(v)) if v.size else 0.0))
            for k, v in self.constraints(r).items()
        }

    def effort(self, r: Rollout) -> float:
        return float(np.mean(np.square(r.torque_ratio)))

    def residuals(self, x: np.ndarray, stage: str) -> np.ndarray:
        """Least-squares residuals of ``stage``; zero when feasible.

        ``stage`` is ``'takeoff'`` or ``'full'``, optionally with ``'@<margin>'``
        overriding the torque margin (continuation from loose to tight torque
        limits). Each group is zero-padded to its size on the full time grid so
        the vector length does not change when the take-off time moves.
        """
        name, _, margin = stage.partition("@")
        r = self.rollout(x, takeoff_only=name == "takeoff")
        n_full = int(round(self.s.duration_s * self.s.sample_rate_hz)) + 1
        nu = self.fb.m.nu

        def pad(v: np.ndarray, size: int) -> np.ndarray:
            out = np.zeros(size)
            out[: v.size] = v[:size]
            return out

        res = []
        for k, v in self.constraints(r, float(margin) if margin else None).items():
            size = n_full * nu if k in ("ineq_torque", "ineq_joint_lo", "ineq_joint_hi") else max(n_full, v.size)
            res.append(pad(v if k.startswith("eq_") else np.minimum(v, 0.0), size if v.size > 3 else v.size))
        if name == "takeoff":
            res.append(np.array([(r.takeoff_com_vel[2] - self.s.takeoff_vz_target_mps) / 0.05]))
            res.append(np.array([(r.takeoff_ang_mom[1] - self.s.takeoff_ang_mom_target_nms) / 0.5]))
            res.append(np.array([(r.takeoff_com_vel[0] - self.s.takeoff_vx_target_mps) / 0.2]))
        res.append(np.array([0.3 * np.sqrt(self.effort(r))]))
        return np.concatenate(res)


# ----------------------------------------------------------------------
# Parallel staged solver.
# ----------------------------------------------------------------------

_WORKER: FlipProblem | None = None


def _init_worker(factory: Callable[[], FlipProblem]) -> None:
    global _WORKER  # noqa: PLW0603  (one problem instance per worker process)
    _WORKER = factory()


def _worker_residuals(args: tuple[np.ndarray, str]) -> np.ndarray:
    assert _WORKER is not None
    return _WORKER.residuals(*args)


def solve_staged(
    factory: Callable[[], FlipProblem],
    x0: np.ndarray,
    stages: Sequence[tuple[str, np.ndarray, int]],
    workers: int,
    log: Callable[[str], None],
    fd_step: float = 1e-4,
) -> np.ndarray:
    """Run ``stages`` of ``(stage_name, free_param_mask, max_evaluations)`` in order.

    Each stage solves a bounded least-squares problem over the masked
    parameters (the others stay fixed) with a parallel forward-difference
    Jacobian. Returns the final parameter vector.
    """
    problem = factory()
    lo, hi = problem.bounds()
    x = np.clip(np.asarray(x0, np.float64), lo + 1e-6, hi - 1e-6)
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(factory,)) as pool:
        for name, mask, max_nfev in stages:
            idx = np.flatnonzero(mask)
            cache: dict[bytes, np.ndarray] = {}

            def full(z: np.ndarray, base: np.ndarray = x, idx: np.ndarray = idx) -> np.ndarray:
                out = base.copy()
                out[idx] = z
                return out

            def fun(z: np.ndarray, name: str = name) -> np.ndarray:
                key = z.tobytes()
                if key not in cache:
                    cache.clear()
                    cache[key] = problem.residuals(full(z), name)
                return cache[key]

            def jac(z: np.ndarray, name: str = name) -> np.ndarray:
                f0 = fun(z)
                pts = [(full(z + fd_step * e), name) for e in np.eye(len(z))]
                cols = list(pool.map(_worker_residuals, pts))
                return (np.array(cols) - f0).T / fd_step

            res = least_squares(fun, x[idx], jac=jac, bounds=(lo[idx], hi[idx]), max_nfev=max_nfev, x_scale="jac")
            x = full(res.x)
            r = fun(res.x)
            log(f"[{name}] t={time.monotonic() - t0:.0f}s cost={0.5 * float(r @ r):.4f} "
                f"max|res|={float(np.max(np.abs(r[:-1]))) if r.size > 1 else 0.0:.3f} "
                f"takeoff={x[-1]:.3f}s ({res.message})")
    return x


def to_clip(
    problem: FlipProblem,
    r: Rollout,
    joint_names: Sequence[str],
    site_names: Sequence[str],
    licence: str,
    source: str,
) -> reference.ReferenceClip:
    """Reference clip (root + joints at the sample rate, FK sites, contact flags)."""
    fb = problem.fb
    site_ids = [fb.m.site(sname).id for sname in site_names]
    sites = []
    for q in r.qpos:
        fb.kinematics(q)
        sites.append(fb.d.site_xpos[site_ids].copy())
    clip = reference.ReferenceClip(
        fps_hz=problem.s.sample_rate_hz,
        qpos=r.qpos.copy(),
        qvel=r.qvel.copy(),
        joint_names=tuple(joint_names),
        site_names=tuple(site_names),
        site_pos=np.array(sites),
        licence=licence,
        source=source,
        foot_contact=r.contact.copy(),
    )
    clip.validate()
    return clip


def _worker_cost(args: tuple[np.ndarray, str]) -> float:
    r = _worker_residuals(args)
    return float(r @ r) if np.all(np.isfinite(r)) else 1e12


def cmaes(
    factory: Callable[[], FlipProblem],
    x0: np.ndarray,
    stage: str,
    sigma0: float,
    generations: int,
    workers: int,
    log: Callable[[str], None],
    popsize: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Minimise ``|residuals(x, stage)|^2`` with CMA-ES (Hansen 2016, rank-mu + rank-1 update).

    Derivative-free, so it tolerates the discontinuities the event times
    introduce (samples switching between stance and flight). Parameters are
    normalised to the box ``bounds()`` and candidates are clipped into it.
    Returns the best parameters seen.
    """
    problem = factory()
    lo, hi = problem.bounds()
    span = hi - lo
    n = len(x0)
    lam = popsize or max(4 + int(3 * np.log(n)), workers)
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
    rng = np.random.default_rng(seed)
    mean = (np.clip(x0, lo, hi) - lo) / span
    sigma, cov = sigma0, np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    best_x, best_f = np.asarray(x0, np.float64), np.inf
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(factory,)) as pool:
        for gen in range(generations):
            evals, evecs = np.linalg.eigh(cov)
            evals = np.maximum(evals, 1e-20)
            z = rng.standard_normal((lam, n))
            y = z @ (evecs * np.sqrt(evals)).T
            cand = np.clip(mean + sigma * y, 0.0, 1.0)
            xs = lo + cand * span
            f = np.array(list(pool.map(_worker_cost, [(x, stage) for x in xs])))
            order = np.argsort(f)
            if f[order[0]] < best_f:
                best_f, best_x = float(f[order[0]]), xs[order[0]].copy()
            y_sel = (cand[order[:mu]] - mean) / sigma
            y_w = weights @ y_sel
            mean = mean + sigma * y_w
            inv_sqrt = evecs @ np.diag(1 / np.sqrt(evals)) @ evecs.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt @ y_w
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chi_n < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
            cov = (
                (1 - c1 - cmu) * cov
                + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * cov)
                + cmu * (y_sel.T * weights) @ y_sel
            )
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
            if gen % 25 == 0 or gen == generations - 1:
                log(
                    f"[cmaes {stage}] gen {gen} t={time.monotonic() - t0:.0f}s best_cost={best_f:.4f} "
                    f"gen_median={np.median(f):.2f} sigma={sigma:.4f}"
                )
    return best_x
