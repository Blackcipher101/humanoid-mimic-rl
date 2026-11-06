"""Join two G1 clips into one "routine" clip: backflip -> transition -> spin kick -> final stand (numpy + mujoco).

Inputs are clips produced by ``scripts/make_g1_cmu_clip.py`` (50 Hz, ``foot_contact`` flags). Steps:

1. **Trim** each source to its move: flight runs (``foot_contact`` false, bounded by contact on both sides)
   give take-off / touch-down; the clip keeps ``LEAD_S`` before the first take-off and ``TAIL_S`` after the last
   touch-down. The cut frames are snapped (within ``SNAP_S``) to the frame whose feet are lowest, so the joins
   do not start from a foot swing.
2. **Windows** (in source time, mapped to routine time later): flip = from the peak forward lean before take-off
   (rotation counted from there) to ``FLIP_WINDOW_AFTER_LANDING_S`` after the touch-down; kick = from
   ``KICK_WINDOW_BEFORE_TAKEOFF_S`` before the first take-off to the last touch-down.
3. **Anchor** the second clip: rotate about world z and translate so its first-frame root xy and heading equal
   the first clip's last frame (no teleport).
4. **Transition** (``TRANSITION_S``): quintic Hermite (zero acceleration at both ends, boundary velocities of the
   two clips scaled by BOUNDARY_VELOCITY_SCALE, zero velocity at the stand) of the joint angles and root position through the G1 home pose
   (standing halfway between the two stances), root
   orientation slerped through upright, then every frame is projected by the retargeting IK
   (``G1Rig.solve``) with feet locks: the feet move from their pose at the end of clip 1 to a planted home stance
   (flat on z = 0, static for ``PLANT_FRACTION`` of the transition), then step to the planted stance at the start of
   clip 2 (``STEP_LIFT_M`` bump). While planted the sole corners are static, so there is no slip.
5. **Final stand**: the last frame's feet stay locked while the joints blend to home (``STAND_BLEND_S``) and are
   then held (``STAND_HOLD_S``).
``qvel`` / ``site_pos`` come from ``g1_retarget.to_clip`` (finite differences of the final ``qpos`` at 50 Hz).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

from humanoid_mimic_rl import g1_model, g1_retarget, metrics, reference

LEAD_S = 0.5  # kept before the first take-off
TAIL_S = 0.8  # kept after the last touch-down
SNAP_S = 0.2  # a cut frame may move this far to find a low-feet frame
FLIP_LEAN_SEARCH_S = 0.6  # the forward-lean peak is searched this far before take-off
FLIP_WINDOW_AFTER_LANDING_S = 0.2
KICK_WINDOW_BEFORE_TAKEOFF_S = 0.3
MIN_FLIGHT_FRAMES = 5
TRANSITION_S = 1.0
HOME_FRACTION = 0.45  # transition fraction at which the pose is the home stand
PLANT_START, PLANT_END = 0.35, 0.6  # transition fractions during which the feet are planted at the home stance
STEP_LIFT_M = 0.04
FOOT_LOCK_WEIGHT = 5.0  # multiplies the IK foot lock weight (W_LOCK) so planted soles stay put
LOCK_RAMP_U = 0.2  # fraction of a foot move over which the lock weight ramps between soft and strong
BOUNDARY_VELOCITY_SCALE = 0.3  # fraction of the clips' junction velocity the transition's Hermite starts / ends with
XY_DONE_FRACTION = 0.8  # fraction of a foot move after which only the height still changes
STAND_BLEND_S = 0.6
STAND_HOLD_S = 0.4
IK_PRIOR_WEIGHT = g1_retarget.W_SMOOTH
SNAP_FOOT_BUCKET_M = 0.02  # foot heights within a bucket are equally good when snapping cuts
SNAP_HEIGHT_BUCKET_M = 0.03  # same for the root height deviation from the G1 home height


@dataclasses.dataclass(frozen=True)
class Trimmed:
    """A source trimmed to its move; times are seconds from the trimmed clip's first frame."""

    qpos: np.ndarray
    first_takeoff_s: float
    last_touchdown_s: float
    window_s: tuple[float, float]
    src_first_frame: int


def flight_runs(foot_contact: np.ndarray) -> list[tuple[int, int]]:
    """Flight runs ``(first air frame, first contact frame after)`` bounded by contact on both sides."""
    air = ~np.asarray(foot_contact, dtype=bool)
    runs, k, n = [], 0, len(air)
    while k < n:
        if not air[k]:
            k += 1
            continue
        a = k
        while k < n and air[k]:
            k += 1
        if a > 0 and k < n and k - a >= MIN_FLIGHT_FRAMES:
            runs.append((a, k))
    return runs


def _feet_low(clip: reference.ReferenceClip, k: int) -> float:
    return float(np.max(clip.site_pos[k, :2, 2]))


def _snap(clip: reference.ReferenceClip, k: int, lo: int, hi: int) -> int:
    """Frame in ``[k - SNAP, k + SNAP]`` (clamped to ``[lo, hi]``) with the lowest feet (2 cm buckets), then the
    root height closest to the G1 home height (3 cm buckets), then the slowest root, then the closest to ``k``:
    a join should not start from a foot swing, a deep crouch or a fast move."""
    r = int(round(SNAP_S * clip.fps_hz))
    last = clip.num_frames - 1

    def root_step_m(j: int) -> float:
        return float(np.linalg.norm(clip.qpos[min(j + 1, last), :3] - clip.qpos[max(j - 1, 0), :3]))

    cand = range(max(lo, k - r), min(hi, k + r) + 1)
    def key(j: int) -> tuple[int, int, float, int]:
        height_dev = abs(float(clip.qpos[j, 2]) - g1_model.G1_HOME_ROOT_HEIGHT_M)
        return (int(_feet_low(clip, j) / SNAP_FOOT_BUCKET_M), int(height_dev / SNAP_HEIGHT_BUCKET_M),
                round(root_step_m(j), 3), abs(j - k))

    return min(cand, key=key)


def _lean_peak_frame(qpos: np.ndarray, fps_hz: float, takeoff: int) -> int:
    """Frame of the maximum forward pitch (body y rotation, ``+`` = forward) in the search span before take-off."""
    pitch = metrics.cumulative_pitch_rad(qpos[None, :, 3:7])[0]
    lo = max(0, takeoff - int(round(FLIP_LEAN_SEARCH_S * fps_hz)))
    return lo + int(np.argmax(pitch[lo : takeoff + 1]))


def trim_flip(clip: reference.ReferenceClip) -> Trimmed:
    """Trim the backflip source: the longest flight run is the flip."""
    runs = flight_runs(clip.foot_contact)
    if not runs:
        raise ValueError("flip source has no bounded flight phase")
    a, b = max(runs, key=lambda r: r[1] - r[0])
    fps = clip.fps_hz
    lean = _lean_peak_frame(clip.qpos, fps, a)
    start = max(0, min(a - int(round(LEAD_S * fps)), lean - int(round(0.1 * fps))))
    end = _snap(clip, min(clip.num_frames - 1, b + int(round(TAIL_S * fps))), b, clip.num_frames - 1)
    win_end = (b + FLIP_WINDOW_AFTER_LANDING_S * fps - start) / fps
    return Trimmed(clip.qpos[start : end + 1].copy(), (a - start) / fps, (b - start) / fps,
                   ((lean - start) / fps, win_end), start)


def trim_kick(clip: reference.ReferenceClip) -> Trimmed:
    """Trim the kick source: first take-off to last touch-down of the bounded flight runs."""
    runs = flight_runs(clip.foot_contact)
    if not runs:
        raise ValueError("kick source has no bounded flight phase")
    a, b = runs[0][0], runs[-1][1]
    fps = clip.fps_hz
    start = _snap(clip, max(0, a - int(round(LEAD_S * fps))), 0, a - 1)
    end = _snap(clip, min(clip.num_frames - 1, b + int(round(TAIL_S * fps))), b, clip.num_frames - 1)
    win_start = max(0.0, a / fps - KICK_WINDOW_BEFORE_TAKEOFF_S)
    return Trimmed(clip.qpos[start : end + 1].copy(), (a - start) / fps, (b - start) / fps,
                   (win_start - start / fps, (b - start) / fps), start)


# ----------------------------------------------------------------------
# Geometry helpers.
# ----------------------------------------------------------------------


def _rot(q_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def _wxyz(r: Rotation) -> np.ndarray:
    x = r.as_quat()
    return np.array([x[3], x[0], x[1], x[2]])


def heading_rad(q_wxyz: np.ndarray) -> float:
    """Yaw of the body x axis projected on the floor (rad about world z)."""
    m = _rot(q_wxyz).as_matrix()
    return float(np.arctan2(m[1, 0], m[0, 0]))


def anchor(qpos: np.ndarray, target_xy: np.ndarray, target_heading_rad: float) -> np.ndarray:
    """Rotate about world z and translate so frame 0 has root xy ``target_xy`` and heading ``target_heading_rad``."""
    out = qpos.copy()
    rz = Rotation.from_euler("z", target_heading_rad - heading_rad(qpos[0, 3:7]))
    pivot = qpos[0, :3].copy()
    out[:, :3] = rz.apply(qpos[:, :3] - pivot) + np.array([target_xy[0], target_xy[1], pivot[2]])
    out[:, 3:7] = np.array([_wxyz(rz * _rot(q)) for q in qpos[:, 3:7]])
    return out


def smootherstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u**3 * (u * (6.0 * u - 15.0) + 10.0)


def hermite5(x0: np.ndarray, v0: np.ndarray, x1: np.ndarray, v1: np.ndarray, duration_s: float, u: float) -> np.ndarray:
    """Quintic Hermite, zero acceleration at both ends; ``u`` in [0, 1], velocities in unit/s."""
    h00 = 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
    h10 = u - 6 * u**3 + 8 * u**4 - 3 * u**5
    h01 = 10 * u**3 - 15 * u**4 + 6 * u**5
    h11 = -4 * u**3 + 7 * u**4 - 3 * u**5
    return h00 * x0 + h10 * duration_s * v0 + h01 * x1 + h11 * duration_s * v1


def _directions(rig: g1_retarget.G1Rig) -> dict[str, np.ndarray]:
    """IK segment direction targets of the pose currently in ``rig`` (so they act as a prior, not a pull)."""
    out = {}
    for name, a, b, _ in g1_retarget.SEGMENTS:
        d = rig.point(rig._g1(b))[0] - rig.point(rig._g1(a))[0]  # noqa: SLF001
        out[f"u_{name}"] = d / np.linalg.norm(d)
    return out


def project_pose(
    rig: g1_retarget.G1Rig, q_prior: np.ndarray, q0: np.ndarray, locks: list[tuple], iters: int = g1_retarget.IK_MAX_ITERS
) -> np.ndarray:
    """The pose closest to ``q_prior`` (joints, root, limb directions, orientations) that satisfies ``locks``."""
    rig.forward(q_prior)
    human = _directions(rig)
    pelvis_rot = rig.body_rot(rig.pelvis_id)
    torso_rot = rig.body_rot(rig.torso_id)
    root_target = rig.point("hipmid")[0]
    return rig.solve(
        q0, human, pelvis_rot, torso_rot, root_target, g1_retarget.W_ROOT_GROUND, locks, q_prior, IK_PRIOR_WEIGHT,
        iters=iters,
    )


def foot_corners_world(rig: g1_retarget.G1Rig, q: np.ndarray) -> dict[str, np.ndarray]:
    """``{"l": (4, 3), "r": (4, 3)}`` world sole corners at ``q``."""
    rig.forward(q)
    return {s: np.array([p for p, _ in rig.foot_corners(s)]) for s in "lr"}


@dataclasses.dataclass(frozen=True)
class FootPose:
    """World pose of a foot body (the ankle-roll link) of one side."""

    pos: np.ndarray  # (3,)
    rot: Rotation


def foot_poses(rig: g1_retarget.G1Rig, q: np.ndarray) -> dict[str, FootPose]:
    rig.forward(q)
    out = {}
    for s in "lr":
        body = rig.points[f"{s}ankle"][0]
        out[s] = FootPose(rig.d.xpos[body].copy(), Rotation.from_matrix(rig.body_rot(body)))
    return out


def flat_foot_pose(rig: g1_retarget.G1Rig, pose: FootPose, side: str) -> FootPose:
    """Same ankle xy and heading, rotated flat, lowered so the four sole corners lie on z = 0."""
    yaw = Rotation.from_euler("z", float(np.arctan2(pose.rot.as_matrix()[1, 0], pose.rot.as_matrix()[0, 0])))
    local = np.array([rig.points[f"{side}corner{i}"][1] for i in range(4)])
    z = -float(np.min((yaw.apply(local))[:, 2]))
    return FootPose(np.array([pose.pos[0], pose.pos[1], z]), yaw)


def corners_of(rig: g1_retarget.G1Rig, pose: FootPose, side: str) -> np.ndarray:
    local = np.array([rig.points[f"{side}corner{i}"][1] for i in range(4)])
    return pose.pos + pose.rot.apply(local)


def home_pose_at(rig: g1_retarget.G1Rig, xy: np.ndarray, heading: float) -> np.ndarray:
    q = rig.home_q.copy()
    q[0:2] = xy
    q[3:7] = _wxyz(Rotation.from_euler("z", heading))
    return q


# ----------------------------------------------------------------------
# Transition and final stand.
# ----------------------------------------------------------------------


def build_transition(
    rig: g1_retarget.G1Rig, q_end: np.ndarray, v_end: np.ndarray, q_start: np.ndarray, v_start: np.ndarray, fps_hz: float
) -> np.ndarray:
    """Frames strictly between ``q_end`` (last frame of clip 1) and ``q_start`` (first frame of clip 2, anchored).

    ``v_*`` are the clips' ``_hermite_state`` velocities in unit/s at the junction. Returns ``(n - 1, nq)`` with
    ``n = round(TRANSITION_S * fps_hz)`` (so ``q_start`` is ``n`` frames after ``q_end``).
    """
    n = int(round(TRANSITION_S * fps_hz))
    duration = n / fps_hz
    heading = heading_rad(q_end[3:7])
    f_a, f_b = foot_poses(rig, q_end), foot_poses(rig, q_start)
    mid_a = 0.5 * (f_a["l"].pos[:2] + f_a["r"].pos[:2])
    mid_b = 0.5 * (f_b["l"].pos[:2] + f_b["r"].pos[:2])
    q_home = home_pose_at(rig, 0.5 * (mid_a + mid_b), heading)  # stand halfway between the two stances
    f_h = {s: flat_foot_pose(rig, p, s) for s, p in foot_poses(rig, q_home).items()}
    r_a, r_h, r_b = _rot(q_end[3:7]), _rot(q_home[3:7]), _rot(q_start[3:7])
    t_home = HOME_FRACTION * duration
    out = []
    q_prev = q_end
    for i in range(1, n):
        tau = i / n
        if tau <= HOME_FRACTION:
            u = tau / HOME_FRACTION
            x = hermite5(_hermite_state(q_end), v_end, _hermite_state(q_home), np.zeros_like(v_end), t_home, u)
            rot = _wxyz(_slerp(r_a, r_h, smootherstep(u)))
        else:
            u = (tau - HOME_FRACTION) / (1.0 - HOME_FRACTION)
            x = hermite5(_hermite_state(q_home), np.zeros_like(v_start), _hermite_state(q_start), v_start,
                         duration - t_home, u)
            rot = _wxyz(_slerp(r_h, r_b, smootherstep(u)))
        q_prior = np.concatenate([[x[-2], x[-1], x[-3]], rot, x[:-3]])
        locks = []
        for side in "lr":
            if tau <= PLANT_START:
                pose = _foot_path(f_a[side], f_h[side], tau / PLANT_START, STEP_LIFT_M)
                weight = _lock_weight(tau / PLANT_START)
            elif tau <= PLANT_END:
                pose, weight = f_h[side], FOOT_LOCK_WEIGHT
            else:
                move = (tau - PLANT_END) / (1.0 - PLANT_END)
                pose = _foot_path(f_h[side], f_b[side], move, STEP_LIFT_M)
                weight = _lock_weight(move)
            target = corners_of(rig, pose, side)
            locks.append(("foot", side, target, np.full(3, weight)))
        q = project_pose(rig, q_prior, q_prev, locks)
        out.append(q)
        q_prev = q
    return np.array(out)


def _lock_weight(u: float) -> float:
    """Foot lock weight multiplier over a foot move: strong at both ends (the foot is within a centimetre of the
    floor and must not slide), soft in the middle so a fast swing does not drag the root."""
    ends = 1.0 - smootherstep(min(u / LOCK_RAMP_U, 1.0)) * smootherstep(min((1.0 - u) / LOCK_RAMP_U, 1.0))
    return 1.0 + (FOOT_LOCK_WEIGHT - 1.0) * ends


def _foot_path(p0: FootPose, p1: FootPose, u: float, lift_m: float) -> FootPose:
    """Foot pose at fraction ``u`` of a move from ``p0`` to ``p1`` (rigid: position and slerped orientation).

    xy follows a smootherstep that completes at ``u = XY_DONE_FRACTION``; the height follows a smootherstep
    plus a ``lift_m * sin(pi u)`` bump. A sole corner is within a centimetre of the floor only near the ends
    of the move, where xy has barely started or is already done, so planted-foot slip stays below 1 cm.
    """
    s_xy = smootherstep(min(u / XY_DONE_FRACTION, 1.0))
    s_z = smootherstep(u)
    pos = np.array([
        (1 - s_xy) * p0.pos[0] + s_xy * p1.pos[0],
        (1 - s_xy) * p0.pos[1] + s_xy * p1.pos[1],
        (1 - s_z) * p0.pos[2] + s_z * p1.pos[2] + lift_m * np.sin(np.pi * u),
    ])
    return FootPose(pos, _slerp(p0.rot, p1.rot, s_z))


def _slerp(a: Rotation, b: Rotation, s: float) -> Rotation:
    delta = (a.inv() * b).as_rotvec()
    return a * Rotation.from_rotvec(s * delta)


def build_final_stand(rig: g1_retarget.G1Rig, q_last: np.ndarray, fps_hz: float) -> np.ndarray:
    """Frames after ``q_last``: joints blend to home with the feet locked where they are, then hold."""
    n_blend = int(round(STAND_BLEND_S * fps_hz))
    n_hold = int(round(STAND_HOLD_S * fps_hz))
    corners = {s: corners_of(rig, flat_foot_pose(rig, p, s), s) for s, p in foot_poses(rig, q_last).items()}
    heading = heading_rad(q_last[3:7])
    q_home = home_pose_at(rig, q_last[:2], heading)
    r0, r1 = _rot(q_last[3:7]), _rot(q_home[3:7])
    out = []
    q_prev = q_last
    for i in range(1, n_blend + 1):
        s = smootherstep(i / n_blend)
        q_prior = np.concatenate([
            (1 - s) * q_last[:3] + s * np.array([q_last[0], q_last[1], q_home[2]]),
            _wxyz(_slerp(r0, r1, s)),
            (1 - s) * q_last[7:] + s * q_home[7:],
        ])
        q = project_pose(rig, q_prior, q_prev, [("foot", side, corners[side], np.full(3, FOOT_LOCK_WEIGHT)) for side in "lr"])
        out.append(q)
        q_prev = q
    out.extend([q_prev.copy() for _ in range(n_hold)])
    return np.array(out)


# ----------------------------------------------------------------------
# Assembly.
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Routine:
    qpos: np.ndarray
    fps_hz: float
    flip_window_s: tuple[float, float]
    kick_window_s: tuple[float, float]
    segments: dict[str, tuple[float, float]]  # name -> (start_s, end_s) in routine time
    sources: dict[str, dict[str, object]]
    transition_frames: tuple[int, int]  # frame range of the synthesized transition
    stand_frames: tuple[int, int]


def _hermite_state(q: np.ndarray) -> np.ndarray:
    """The interpolated part of a pose: joints, root z, root x, root y."""
    return np.concatenate([q[7:], [q[2], q[0], q[1]]])


def _boundary_velocity(qpos: np.ndarray, fps_hz: float, at_end: bool) -> np.ndarray:
    """Velocity (unit/s) of ``_hermite_state`` at a clip's last / first frame by a one-sided difference."""
    a, b = (qpos[-2], qpos[-1]) if at_end else (qpos[0], qpos[1])
    return (_hermite_state(b) - _hermite_state(a)) * fps_hz


def assemble(
    rig: g1_retarget.G1Rig, flip: reference.ReferenceClip, kick: reference.ReferenceClip
) -> Routine:
    """Trim, anchor and join; see the module docstring."""
    if flip.fps_hz != kick.fps_hz:
        raise ValueError(f"source frame rates differ: {flip.fps_hz} vs {kick.fps_hz}")
    fps = flip.fps_hz
    a, b = trim_flip(flip), trim_kick(kick)
    q_a_end = a.qpos[-1]
    b_qpos = anchor(b.qpos, q_a_end[:2], heading_rad(q_a_end[3:7]))
    v_end = BOUNDARY_VELOCITY_SCALE * _boundary_velocity(a.qpos, fps, True)
    v_start = BOUNDARY_VELOCITY_SCALE * _boundary_velocity(b_qpos, fps, False)
    trans = build_transition(rig, q_a_end, v_end, b_qpos[0], v_start, fps)
    stand = build_final_stand(rig, b_qpos[-1], fps)
    qpos = np.concatenate([a.qpos, trans, b_qpos, stand])
    for k in range(1, len(qpos)):  # sign-continuous quaternions
        if np.dot(qpos[k, 3:7], qpos[k - 1, 3:7]) < 0.0:
            qpos[k, 3:7] = -qpos[k, 3:7]
    n_a, n_t, n_b = len(a.qpos), len(trans), len(b_qpos)
    off_b = (n_a + n_t) / fps
    flip_w = a.window_s
    kick_w = (off_b + b.window_s[0], off_b + b.window_s[1])
    segs = {
        "backflip": (0.0, (n_a - 1) / fps),
        "transition": ((n_a - 1) / fps, (n_a + n_t) / fps),
        "spin_kick": (off_b, off_b + (n_b - 1) / fps),
        "final_stand": (off_b + (n_b - 1) / fps, (len(qpos) - 1) / fps),
    }
    sources = {
        "backflip": {"src_first_frame": a.src_first_frame, "src_frames": len(a.qpos),
                     "takeoff_s": a.first_takeoff_s, "touchdown_s": a.last_touchdown_s},
        "spin_kick": {"src_first_frame": b.src_first_frame, "src_frames": len(b.qpos),
                      "takeoff_s": off_b + b.first_takeoff_s, "touchdown_s": off_b + b.last_touchdown_s},
    }
    return Routine(qpos, fps, flip_w, kick_w, segs, sources, (n_a, n_a + n_t), (n_a + n_t + n_b, len(qpos)))


# ----------------------------------------------------------------------
# Continuity checks (reported and unit-tested).
# ----------------------------------------------------------------------

CONTACT_Z_M = 0.01  # a sole corner this close to the floor counts as in contact for the slip check


def _runs_true(mask: np.ndarray) -> list[tuple[int, int]]:
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            a = k
            while k < len(mask) and mask[k]:
                k += 1
            out.append((a, k))
        else:
            k += 1
    return out


def continuity_report(rig: g1_retarget.G1Rig, routine: Routine) -> dict[str, object]:
    """Root step sizes (cm / frame), foot slip of planted feet (cm) in the synthesized frames, floor penetration.

    Steps are given for the whole clip (the source flight / landing frames dominate it) and for the
    synthesized spans: transition (plus the two junction frames on each side) and final stand.
    ``max_planted_foot_slip_cm``: for every run of >= 3 frames inside the synthesized frames in which a foot
    has all four sole corners within ``CONTACT_Z_M`` of the floor, the largest xy distance of a corner from its
    mean over the run.
    """
    q = routine.qpos
    step = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)  # step[k] = |p[k+1] - p[k]|
    t0, t1 = routine.transition_frames
    s0, s1 = routine.stand_frames
    lo, hi = max(t0 - 2, 0), min(t1 + 2, len(q) - 1)
    corners = [foot_corners_world(rig, row) for row in q]
    slip = 0.0
    for side in "lr":
        arr = np.array([c[side] for c in corners])  # (T, 4, 3)
        planted = np.all(arr[:, :, 2] < CONTACT_Z_M, axis=1)
        synth = np.zeros(len(q), dtype=bool)
        synth[t0:t1] = True
        synth[s0 - 2 : s1] = True
        for a, b in _runs_true(planted & synth):
            if b - a >= 3:
                xy = arr[a:b, :, :2]
                slip = max(slip, float(np.linalg.norm(xy - xy.mean(axis=0), axis=-1).max()))
    clearance = min(g1_retarget.floor_clearance_m(rig, row) for row in np.concatenate([q[t0:t1], q[s0:s1]]))
    return {
        "max_root_step_cm_whole_clip": 100.0 * float(step.max()),
        "max_root_step_cm_whole_clip_at_frame": int(np.argmax(step)),
        "max_root_step_cm_transition": 100.0 * float(step[lo:hi].max()),
        "max_root_step_cm_final_stand": 100.0 * float(step[max(s0 - 2, 0) :].max()),
        "max_planted_foot_slip_cm": 100.0 * slip,
        "min_floor_clearance_m_synthesized_frames": float(clearance),
        "max_joint_step_rad_transition": float(np.abs(np.diff(q[lo : hi + 1, 7:], axis=0)).max()),
    }
