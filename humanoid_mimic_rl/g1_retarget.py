"""Retarget CMU mocap clips (``cmu_mocap``) to the Unitree G1 29-DoF model (mujoco + scipy, no JAX).

Pipeline (all world frames z-up, SI units, floor z = 0):

1. human key points (``human_keypoints``) from the ASF/AMC forward kinematics, yaw-aligned so frame 0
   starts at the origin facing +x, low-pass filtered, then resampled with **Froude time scaling**:
   lengths scale by ``s`` = G1 leg length / human leg length, so times scale by ``sqrt(s)`` (same
   ``g``; accelerations, hence ballistic flights, are preserved);
2. foot / hand ground contacts of the human (``detect_contacts``);
3. pass 1: per-frame damped least-squares IK on the G1 (joint limits as bounds): segment *directions*
   (thigh, shin, upper arm, forearm, spine, hip/shoulder lines, ...) match the human's (scale free,
   so differing limb proportions cannot pull the body apart), pelvis/torso orientation, and a root
   position target (``s`` x the human hip-centre);
4. pass 2: contact locks from pass 1 (feet: four sole corners flat on z = 0 and static; hands: both
   capsule end points on the floor and static), floor-penetration hinge terms for every floor collider,
   smoothness prior; then a zero-phase low-pass, and a final lock re-projection pass.

Every IK step is exact in the G1 kinematics (``mj_jac``), the joint limits are hard bounds.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from humanoid_mimic_rl import cmu_mocap, g1_backflip, g1_model, quat, reference

KEYPOINT_NAMES: tuple[str, ...] = (
    "hipmid", "lhip", "rhip", "lknee", "rknee", "lankle", "rankle", "ltoe", "rtoe",
    "lsh", "rsh", "lelb", "relb", "lwri", "rwri", "lpalm", "rpalm", "headtop", "spine_low", "neck_base",
)  # fmt: skip
CLIP_RATE_HZ = 50.0
HUMAN_LOWPASS_HZ = 10.0  # on the human key points (native rate), before resampling
OUTPUT_LOWPASS_HZ = 6.0  # on the G1 qpos (50 Hz), second differences feed the torque check
HAND_RADIUS_M = 0.05  # left/right_hand_collision capsule radius (G1 model)

# Human contact detection (m, m/s).
FOOT_CONTACT_ANKLE_RISE_M = 0.04  # ankle this close to its stance height
HAND_CONTACT_WRIST_Z_M = 0.20  # wrist joint below this = palm on the floor
CONTACT_MAX_SPEED_MPS = 1.5
MIN_CONTACT_FRAMES = 3
MAX_GAP_FRAMES = 3

EE_NAMES: tuple[str, ...] = ("lfoot", "rfoot", "lhand", "rhand")

# IK segments: (name, start, end, weight); residual = weight * L_g1 * (u_g1 - u_human).
SEGMENTS: tuple[tuple[str, str, str, float], ...] = (
    ("thighL", "lhip", "lknee", 1.0), ("shinL", "lknee", "lankle", 1.0), ("footL", "lankle", "ltoe", 0.3),
    ("thighR", "rhip", "rknee", 1.0), ("shinR", "rknee", "rankle", 1.0), ("footR", "rankle", "rtoe", 0.3),
    ("hipline", "rhip", "lhip", 1.0), ("spine", "hipmid", "shmid", 1.0), ("shline", "rsh", "lsh", 1.0),
    ("uarmL", "lsh", "lelb", 1.0), ("farmL", "lelb", "lwri", 1.0), ("handL", "lwri", "lpalm", 0.3),
    ("uarmR", "rsh", "relb", 1.0), ("farmR", "relb", "rwri", 1.0), ("handR", "rwri", "rpalm", 0.3),
    ("neck", "shmid", "headtop", 0.3),
)  # fmt: skip
W_ORIENT = 0.3  # m / rad, pelvis and torso orientation terms
W_ROOT_GROUND = 0.3  # m / m, root position target while a contact is locked
W_ROOT_FLIGHT = 3.0
W_LOCK = 200.0
LOCK_RAMP_FRAMES = 6  # lock weight smoothsteps in/out around a contact run (no impact-like snaps)
W_FLOOR = 100.0
W_SMOOTH = 50.0  # weight of the pull towards the low-passed trajectory (per m / rad), >> tracking weights
SMOOTH_ITERATIONS = 3
LM_DAMPING = 1e-3
IK_MAX_ITERS = 60
IK_STEP_LIMIT = 0.15  # trust region per iteration (m / rad): stops the stiff lock terms from flipping limbs
IK_STEP_TOL = 1e-5


def lowpass(x: np.ndarray, fps_hz: float, fc_hz: float, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0 (reflect-padded); ``x`` is returned unchanged if fc >= Nyquist."""
    if fc_hz >= 0.5 * fps_hz:
        return x
    b, a = butter(order, fc_hz / (0.5 * fps_hz))
    pad = min(3 * max(len(a), len(b)) * 4, x.shape[0] - 1)
    return filtfilt(b, a, x, axis=0, padlen=pad)


def _frame_from_axes(y_axis: np.ndarray, z_hint: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(T, 3, 3)`` (columns x fwd, y left, z up) from a left axis and an up hint."""
    y = y_axis / np.maximum(np.linalg.norm(y_axis, axis=-1, keepdims=True), 1e-9)
    z = z_hint - np.sum(z_hint * y, axis=-1, keepdims=True) * y
    z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-9)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


@dataclasses.dataclass(frozen=True)
class HumanMotion:
    """Human key points (m, z-up, yaw-aligned) at ``fps_hz`` plus pelvis / torso frames."""

    fps_hz: float
    points: dict[str, np.ndarray]  # name -> (T, 3)
    pelvis_rot: np.ndarray  # (T, 3, 3) world <- pelvis (x fwd, y left, z up)
    torso_rot: np.ndarray
    leg_length_m: float  # mean hip -> knee -> ankle path length of both legs (frame 0)

    @property
    def num_frames(self) -> int:
        return int(self.points["hipmid"].shape[0])


def human_keypoints(skeleton: cmu_mocap.Skeleton, motion: cmu_mocap.Motion, fps_hz: float) -> HumanMotion:
    """Key points of a CMU motion in metres, z-up, frame 0 at the origin facing +x."""
    pose = cmu_mocap.forward_kinematics(skeleton, motion)
    z = cmu_mocap.to_z_up
    p = {
        "lhip": z(pose.origin["lfemur"]), "rhip": z(pose.origin["rfemur"]),
        "lknee": z(pose.origin["ltibia"]), "rknee": z(pose.origin["rtibia"]),
        "lankle": z(pose.origin["lfoot"]), "rankle": z(pose.origin["rfoot"]),
        "ltoe": z(pose.tip["ltoes"]), "rtoe": z(pose.tip["rtoes"]),
        "lsh": z(pose.origin["lhumerus"]), "rsh": z(pose.origin["rhumerus"]),
        "lelb": z(pose.origin["lradius"]), "relb": z(pose.origin["rradius"]),
        "lwri": z(pose.origin["lwrist"]), "rwri": z(pose.origin["rwrist"]),
        "lpalm": z(pose.tip["lhand"]), "rpalm": z(pose.tip["rhand"]),
        "headtop": z(pose.tip["head"]), "spine_low": z(pose.origin["upperback"]),
        "neck_base": z(pose.origin["lowerneck"]),
    }  # fmt: skip
    return keypoints_to_motion(p, fps_hz)


def keypoints_to_motion(p: dict[str, np.ndarray], fps_hz: float) -> HumanMotion:
    """``HumanMotion`` from raw key points (m, z-up, every name in ``KEYPOINT_NAMES`` except ``hipmid``):
    low-pass, frame 0 moved to the origin facing +x, pelvis / torso frames, leg length."""
    missing = set(KEYPOINT_NAMES) - {"hipmid"} - set(p)
    if missing:
        raise ValueError(f"missing key points: {sorted(missing)}")
    p = dict(p)
    p["hipmid"] = 0.5 * (p["lhip"] + p["rhip"])
    p = {k: lowpass(v, fps_hz, HUMAN_LOWPASS_HZ) for k, v in p.items()}
    pelvis = _frame_from_axes(p["lhip"] - p["rhip"], p["spine_low"] - p["hipmid"])
    yaw = np.arctan2(pelvis[0, 1, 0], pelvis[0, 0, 0])
    rz = Rotation.from_euler("z", -yaw).as_matrix()
    origin = np.array([p["hipmid"][0, 0], p["hipmid"][0, 1], 0.0])
    p = {k: (v - origin) @ rz.T for k, v in p.items()}
    pelvis = _frame_from_axes(p["lhip"] - p["rhip"], p["spine_low"] - p["hipmid"])
    torso = _frame_from_axes(p["lsh"] - p["rsh"], p["neck_base"] - p["spine_low"])
    leg = np.mean(
        [np.linalg.norm(p[f"{s}knee"][0] - p[f"{s}hip"][0]) + np.linalg.norm(p[f"{s}ankle"][0] - p[f"{s}knee"][0])
         for s in "lr"]
    )  # fmt: skip
    return HumanMotion(fps_hz, p, pelvis, torso, float(leg))


def froude_resample(h: HumanMotion, scale: float, out_fps_hz: float = CLIP_RATE_HZ) -> HumanMotion:
    """Scale lengths by ``scale`` and time by ``sqrt(scale)`` (Froude similarity), sampled at ``out_fps_hz``.

    Positions are linearly interpolated, the frames by slerp. The returned motion is in G1 metres / G1 seconds.
    """
    t_h = np.arange(h.num_frames) / h.fps_hz
    t_new = np.arange(0.0, t_h[-1] * np.sqrt(scale) + 1e-9, 1.0 / out_fps_hz)
    tau = t_new / np.sqrt(scale)
    pts = {k: scale * np.stack([np.interp(tau, t_h, v[:, i]) for i in range(3)], axis=1) for k, v in h.points.items()}

    def slerp(rots: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Slerp  # noqa: PLC0415

        return Slerp(t_h, Rotation.from_matrix(rots))(tau).as_matrix()

    return HumanMotion(out_fps_hz, pts, slerp(h.pelvis_rot), slerp(h.torso_rot), h.leg_length_m * scale)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` index runs of True in a 1-D bool array."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def clean_mask(mask: np.ndarray, min_frames: int = MIN_CONTACT_FRAMES, max_gap: int = MAX_GAP_FRAMES) -> np.ndarray:
    """Fill False gaps shorter than ``max_gap`` and drop True runs shorter than ``min_frames``."""
    out = mask.copy()
    for a, b in _runs(~out):
        if a > 0 and b < len(out) and b - a < max_gap:
            out[a:b] = True
    for a, b in _runs(out):
        if b - a < min_frames:
            out[a:b] = False
    return out


def detect_contacts(h: HumanMotion) -> dict[str, np.ndarray]:
    """Per end-effector ground-contact masks ``(T,)`` of the human (``h`` in its own metres, floor z = 0).

    Foot: the ankle is within ``FOOT_CONTACT_ANKLE_RISE_M`` of that foot's stance height (median ankle height
    of its lowest 20 % of frames) and slower than ``CONTACT_MAX_SPEED_MPS``. Hand: wrist joint below
    ``HAND_CONTACT_WRIST_Z_M`` and slower than the limit. Speeds are measured in the motion's own units
    (callers pass the unscaled human motion).
    """
    out: dict[str, np.ndarray] = {}
    for ee, joint in (("lfoot", "lankle"), ("rfoot", "rankle"), ("lhand", "lwri"), ("rhand", "rwri")):
        pos = h.points[joint]
        speed = np.linalg.norm(np.gradient(pos, 1.0 / h.fps_hz, axis=0), axis=1)
        if ee.endswith("foot"):
            low = np.sort(pos[:, 2])[: max(5, h.num_frames // 5)]
            mask = pos[:, 2] < np.median(low) + FOOT_CONTACT_ANKLE_RISE_M
        else:
            mask = pos[:, 2] < HAND_CONTACT_WRIST_Z_M
        out[ee] = clean_mask(mask & (speed < CONTACT_MAX_SPEED_MPS))
    return out


@dataclasses.dataclass(frozen=True)
class Lock:
    """World targets of a static contact: points (n, 3), per-point axis weights (n, 3)."""

    points: str  # key into ``G1Rig.lock_points``
    xyz: np.ndarray  # (n, 3)
    weight: np.ndarray  # (n, 3)


class G1Rig:
    """G1 kinematics with named points, Jacobians (``mj_jac``) and the IK solver."""

    def __init__(self, assets: Mapping[str, bytes]) -> None:
        self.m = g1_backflip.load_model(assets, 1.0)
        self.d = mujoco.MjData(self.m)
        self.nv = self.m.nv
        self.lo = self.m.jnt_range[1:, 0].copy()
        self.hi = self.m.jnt_range[1:, 1].copy()
        bid = lambda n: int(self.m.body(n).id)  # noqa: E731
        self.pelvis_id, self.torso_id = bid("pelvis"), bid("torso_link")
        foot = {"l": bid("left_ankle_roll_link"), "r": bid("right_ankle_roll_link")}
        sx, sy, sz = g1_model.FOOT_SOLE_CENTRE_LOCAL_M
        hx, hy = g1_model.FOOT_HALF_LENGTH_M, g1_model.FOOT_HALF_WIDTH_M
        corners = [np.array([sx + a * hx, sy + b * hy, sz]) for a in (-1, 1) for b in (-1, 1)]
        self.points: dict[str, tuple[int, np.ndarray]] = {}
        for s, side in (("l", "left"), ("r", "right")):
            self.points.update({
                f"{s}hip": (bid(f"{side}_hip_pitch_link"), np.zeros(3)),
                f"{s}knee": (bid(f"{side}_knee_link"), np.zeros(3)),
                f"{s}ankle": (foot[s], np.zeros(3)),
                f"{s}toe": (foot[s], np.array([sx + hx, 0.0, sz])),
                f"{s}sh": (bid(f"{side}_shoulder_pitch_link"), np.zeros(3)),
                f"{s}elb": (bid(f"{side}_elbow_link"), np.zeros(3)),
                f"{s}wri": (bid(f"{side}_wrist_roll_link"), np.zeros(3)),
            })  # fmt: skip
            site = self.m.site(f"{side}_palm")
            self.points[f"{s}palm"] = (int(site.bodyid[0]), np.asarray(site.pos).copy())
            for i, c in enumerate(corners):
                self.points[f"{s}corner{i}"] = (foot[s], c)
        head = self.m.geom("head_collision")
        self.points["headtop"] = (int(head.bodyid[0]), np.asarray(head.pos) + np.array([0.0, 0.0, 0.06]))
        # floor colliders other than the feet: (geom id, radius, end points in the geom frame)
        self.hand_geoms = {"l": int(self.m.geom("left_hand_collision").id), "r": int(self.m.geom("right_hand_collision").id)}
        self.floor_ends: list[tuple[int, float, np.ndarray]] = []
        for name in g1_model.FLOOR_PAIR_GEOMS:
            gid = int(self.m.geom(name).id)
            radius, half = float(self.m.geom_size[gid, 0]), float(self.m.geom_size[gid, 1])
            if self.m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                ends = np.zeros((1, 3))
            else:
                ends = np.array([[0.0, 0.0, half], [0.0, 0.0, -half]])
            self.floor_ends.append((gid, radius, ends))
        self.home_q = self.m.keyframe("home").qpos.copy()
        self.seg_len = {}
        self.forward(self.home_q)
        for name, a, b, _ in SEGMENTS:
            self.seg_len[name] = float(np.linalg.norm(self.point(self._g1(b))[0] - self.point(self._g1(a))[0]))
        self.leg_length_m = float(
            np.mean([np.linalg.norm(self.point(f"{s}knee")[0] - self.point(f"{s}hip")[0])
                     + np.linalg.norm(self.point(f"{s}ankle")[0] - self.point(f"{s}knee")[0]) for s in "lr"])
        )  # fmt: skip

    # -- kinematics -----------------------------------------------------
    def forward(self, q: np.ndarray) -> None:
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    @staticmethod
    def _g1(name: str) -> str:
        return {"hipmid": "hipmid", "shmid": "shmid"}.get(name, name)

    def point(self, name: str, jac: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """World position (and ``(3, nv)`` Jacobian) of a named point; ``hipmid`` / ``shmid`` are midpoints."""
        if name in ("hipmid", "shmid"):
            a, b = ("lhip", "rhip") if name == "hipmid" else ("lsh", "rsh")
            pa, ja = self.point(a, jac)
            pb, jb = self.point(b, jac)
            return 0.5 * (pa + pb), (0.5 * (ja + jb) if jac else None)
        body, local = self.points[name]
        return self._body_point(body, local, jac)

    def _body_point(self, body: int, local: np.ndarray, jac: bool) -> tuple[np.ndarray, np.ndarray | None]:
        p = self.d.xpos[body] + self.d.xmat[body].reshape(3, 3) @ local
        if not jac:
            return p, None
        jp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jp, None, p, body)
        return p, jp

    def geom_ends(self, gid: int, ends: np.ndarray, jac: bool) -> list[tuple[np.ndarray, np.ndarray | None]]:
        body = int(self.m.geom_bodyid[gid])
        rot = self.d.geom_xmat[gid].reshape(3, 3)
        out = []
        for e in ends:
            p = self.d.geom_xpos[gid] + rot @ e
            if jac:
                jp = np.zeros((3, self.nv))
                mujoco.mj_jac(self.m, self.d, jp, None, p, body)
                out.append((p, jp))
            else:
                out.append((p, None))
        return out

    def hand_ends(self, side: str, jac: bool = False) -> list[tuple[np.ndarray, np.ndarray | None]]:
        gid = self.hand_geoms[side]
        half = float(self.m.geom_size[gid, 1])
        return self.geom_ends(gid, np.array([[0.0, 0.0, half], [0.0, 0.0, -half]]), jac)

    def foot_corners(self, side: str, jac: bool = False) -> list[tuple[np.ndarray, np.ndarray | None]]:
        return [self.point(f"{side}corner{i}", jac) for i in range(4)]

    def body_rot(self, body: int) -> np.ndarray:
        return self.d.xmat[body].reshape(3, 3).copy()

    # -- inverse kinematics ----------------------------------------------
    def solve(
        self,
        q0: np.ndarray,
        human: Mapping[str, np.ndarray],
        pelvis_rot: np.ndarray,
        torso_rot: np.ndarray,
        root_target: np.ndarray,
        w_root: float,
        locks: Sequence[tuple[str, str, np.ndarray, np.ndarray]],
        prior_q: np.ndarray | None,
        prior_weight: float = 0.0,
        use_floor: bool = True,
        iters: int = IK_MAX_ITERS,
    ) -> np.ndarray:
        """Damped least-squares IK for one frame, starting from ``q0``.

        ``locks``: ``(kind, side, xyz, weight)`` with kind ``"foot"`` (4 sole corners, ``xyz`` (4, 3)) or ``"hand"``
        (2 capsule end points, ``xyz`` (2, 3)); ``weight`` is per axis (3,). Returns the new ``qpos`` (36,).
        """
        q = q0.copy()
        for _ in range(iters):
            self.forward(q)
            res: list[np.ndarray] = []
            jacs: list[np.ndarray] = []

            def add(r: np.ndarray, j: np.ndarray) -> None:
                res.append(np.atleast_1d(r))
                jacs.append(np.atleast_2d(j))

            for name, a, b, w in SEGMENTS:
                pa, ja = self.point(self._g1(a), True)
                pb, jb = self.point(self._g1(b), True)
                d = pb - pa
                n = np.linalg.norm(d)
                u = d / n
                length = self.seg_len[name]
                uh = human[f"u_{name}"]
                add(w * length * (u - uh), w * length * (np.eye(3) - np.outer(u, u)) @ (jb - ja) / n)
            p, jp = self.point("hipmid", True)
            add(w_root * (p - root_target), w_root * jp)
            for body, rt in ((self.pelvis_id, pelvis_rot), (self.torso_id, torso_rot)):
                jr = np.zeros((3, self.nv))
                mujoco.mj_jacBody(self.m, self.d, None, jr, body)
                err = Rotation.from_matrix(self.body_rot(body) @ rt.T).as_rotvec()
                add(W_ORIENT * err, W_ORIENT * jr)
            if use_floor:
                self._floor_terms(add)
            for kind, side, xyz, weight in locks:
                pts = self.foot_corners(side, True) if kind == "foot" else self.hand_ends(side, True)
                for (pp, jj), target in zip(pts, xyz):
                    add(W_LOCK * weight * (pp - target), W_LOCK * weight[:, None] * jj)
            if prior_q is not None:
                eye = np.eye(self.nv)
                w = prior_weight
                drot = Rotation.from_quat(_wxyz_to_xyzw(prior_q[3:7])).inv() * Rotation.from_quat(_wxyz_to_xyzw(q[3:7]))
                add(w * (q[:3] - prior_q[:3]), w * eye[:3])
                add(w * drot.as_rotvec(), w * eye[3:6])
                add(w * (q[7:] - prior_q[7:]), w * eye[6:])
            r = np.concatenate(res)
            jac = np.vstack(jacs)
            lo = np.full(self.nv, -IK_STEP_LIMIT)
            hi = np.full(self.nv, IK_STEP_LIMIT)
            lo[6:] = np.maximum(self.lo - q[7:], -IK_STEP_LIMIT)
            hi[6:] = np.minimum(self.hi - q[7:], IK_STEP_LIMIT)
            a = np.vstack([jac, np.sqrt(LM_DAMPING) * np.eye(self.nv)])
            b = np.concatenate([-r, np.zeros(self.nv)])
            dv = lsq_linear(a, b, bounds=(lo, hi), method="bvls").x
            mujoco.mj_integratePos(self.m, q, dv, 1.0)
            q[7:] = np.clip(q[7:], self.lo, self.hi)
            if np.max(np.abs(dv)) < IK_STEP_TOL:
                break
        return q

    def _floor_terms(self, add) -> None:  # noqa: ANN001
        """Hinge terms ``max(0, -z)`` for the lowest points of every floor collider (feet: sole corners)."""
        for side in "lr":
            for p, jp in self.foot_corners(side, True):
                if p[2] < 0.0:
                    add(W_FLOOR * np.array([p[2]]), W_FLOOR * jp[2:3])
        for gid, radius, ends in self.floor_ends:
            for p, jp in self.geom_ends(gid, ends, True):
                depth = p[2] - radius
                if depth < 0.0:
                    add(W_FLOOR * np.array([depth]), W_FLOOR * jp[2:3])


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def frame_targets(h: HumanMotion, k: int) -> dict[str, np.ndarray]:
    """Per-frame human segment directions ``u_<segment>`` for ``G1Rig.solve``."""
    pts = dict(h.points)
    pts["shmid"] = 0.5 * (pts["lsh"] + pts["rsh"])
    return {f"u_{name}": _unit(pts[b][k] - pts[a][k]) for name, a, b, _ in SEGMENTS}


@dataclasses.dataclass(frozen=True)
class RetargetResult:
    qpos: np.ndarray  # (T, 36) MuJoCo order, wxyz root quaternion
    fps_hz: float
    scale: float  # G1 leg length / human leg length (length scale; time scale is sqrt)
    human_contacts: dict[str, np.ndarray]  # (T,) per end effector (the retargeted time base)
    human: HumanMotion  # key points scaled to G1 metres/seconds


def _qpos_from_root(pos: np.ndarray, rot: np.ndarray, joints: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rot).as_quat()
    return np.concatenate([pos, [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], joints])


def _make_locks(rig: G1Rig, q_pass1: np.ndarray, contacts: dict[str, np.ndarray]) -> list[list[tuple]]:
    """Per-frame lock lists from the pass-1 poses: for each contact run a static flat foot / static flat hand.

    Foot: mean ankle xy and mean heading of the run, sole flat on z = 0 (four corner targets). Hand: mean xy of
    both capsule end points, both at the capsule radius above the floor (palm flat).
    """
    locks: list[list[tuple]] = [[] for _ in range(q_pass1.shape[0])]
    for ee, mask in contacts.items():
        side = ee[0]
        for a, b in _runs(mask):
            if ee.endswith("foot"):
                yaw, xy = [], []
                for k in range(a, b):
                    rig.forward(q_pass1[k])
                    rot = rig.body_rot(rig.points[f"{side}ankle"][0])
                    yaw.append(np.arctan2(rot[1, 0], rot[0, 0]))
                    xy.append(rig.point(f"{side}ankle")[0][:2])
                rz = Rotation.from_euler("z", float(np.angle(np.mean(np.exp(1j * np.array(yaw)))))).as_matrix()[:2, :2]
                ankle_xy = np.mean(xy, axis=0)
                local = [rig.points[f"{side}corner{i}"][1] for i in range(4)]
                xyz = np.array([[*(ankle_xy + rz @ c[:2]), 0.0] for c in local])
                entry = ("foot", side, xyz, np.ones(3))
            else:
                ends = []
                for k in range(a, b):
                    rig.forward(q_pass1[k])
                    ends.append([p[:2] for p, _ in rig.hand_ends(side)])
                xyz = np.array([[e[0], e[1], HAND_RADIUS_M] for e in np.mean(ends, axis=0)])
                entry = ("hand", side, xyz, np.ones(3))
            kind, side_, xyz_, w_ = entry
            for k in range(max(a - LOCK_RAMP_FRAMES, 0), min(b + LOCK_RAMP_FRAMES, len(locks))):
                dist = max(a - k, k - (b - 1), 0)  # frames outside the run
                ramp = 1.0 - dist / (LOCK_RAMP_FRAMES + 1.0)
                locks[k].append((kind, side_, xyz_, w_ * ramp * ramp * (3.0 - 2.0 * ramp)))
    return locks


def retarget(
    rig: G1Rig,
    skeleton: cmu_mocap.Skeleton,
    motion: cmu_mocap.Motion,
    amc_fps_hz: float,
    log=None,  # noqa: ANN001
    final_lowpass_hz: float | None = None,
) -> RetargetResult:
    """Retarget one CMU motion; see the module docstring."""
    return retarget_human(rig, human_keypoints(skeleton, motion, amc_fps_hz), log, final_lowpass_hz)


def retarget_human(
    rig: G1Rig,
    native: HumanMotion,
    log=None,  # noqa: ANN001
    final_lowpass_hz: float | None = None,
) -> RetargetResult:
    """Retarget human key points at their native rate (``keypoints_to_motion``); see the module docstring."""
    contacts_native = detect_contacts(native)
    scale = rig.leg_length_m / native.leg_length_m
    h = froude_resample(native, scale)
    t_native = np.arange(native.num_frames) / native.fps_hz
    t_out = np.arange(h.num_frames) / h.fps_hz / np.sqrt(scale)
    contacts = {
        ee: clean_mask(np.interp(t_out, t_native, m.astype(float)) > 0.5) for ee, m in contacts_native.items()
    }
    n = h.num_frames
    airborne = ~(contacts["lfoot"] | contacts["rfoot"] | contacts["lhand"] | contacts["rhand"])
    targets = [frame_targets(h, k) for k in range(n)]
    root_t = h.points["hipmid"]

    def run(q_init: np.ndarray, lock_lists, prior, prior_weight: float) -> np.ndarray:
        out = np.zeros((n, rig.m.nq))
        q = q_init
        for k in range(n):
            w_root = W_ROOT_FLIGHT if airborne[k] else W_ROOT_GROUND
            if k == 0:
                q0 = q.copy()
                q0[:3] = root_t[0]
                q0[3:7] = _qpos_from_root(np.zeros(3), h.pelvis_rot[0], np.zeros(0))[3:7]
                iters = 4 * IK_MAX_ITERS
            else:
                q0, iters = out[k - 1].copy(), IK_MAX_ITERS
            pq = None if prior is None else prior[k]
            q = rig.solve(
                q0, targets[k], h.pelvis_rot[k], h.torso_rot[k], root_t[k], w_root,
                lock_lists[k] if lock_lists is not None else [], pq, prior_weight, iters=iters,
            )  # fmt: skip
            out[k] = q
            if log and k % 50 == 0:
                log(f"  frame {k}/{n}")
        return out

    q = run(rig.home_q.copy(), None, None, 0.0)  # pass 1: unlocked tracking
    locks = _make_locks(rig, q, contacts)
    q = run(q[0].copy(), locks, None, 0.0)  # pass 2: contact locks
    for _ in range(SMOOTH_ITERATIONS):  # pass 3+: pull towards the low-passed trajectory, locks stay hard
        smooth = _smooth_qpos(q, h.fps_hz, OUTPUT_LOWPASS_HZ)
        q = run(smooth[0].copy(), locks, smooth, W_SMOOTH)
    q5 = q if final_lowpass_hz is None else _smooth_qpos(q, h.fps_hz, final_lowpass_hz)
    q5[:, 7:] = np.clip(q5[:, 7:], rig.lo, rig.hi)
    return RetargetResult(q5, h.fps_hz, scale, contacts, h)


def _smooth_qpos(qpos: np.ndarray, fps_hz: float, fc_hz: float) -> np.ndarray:
    """Zero-phase low-pass of root position, joint angles and (sign-continuous, renormalised) root quaternion."""
    out = qpos.copy()
    quats = qpos[:, 3:7].copy()
    for k in range(1, len(quats)):
        if np.dot(quats[k], quats[k - 1]) < 0.0:
            quats[k] = -quats[k]
    out[:, 3:7] = quat.quat_normalize(lowpass(quats, fps_hz, fc_hz), np)
    out[:, :3] = lowpass(qpos[:, :3], fps_hz, fc_hz)
    out[:, 7:] = lowpass(qpos[:, 7:], fps_hz, fc_hz)
    return out


def apply_time_scale(qpos: np.ndarray, fps_hz: float, factor: float) -> np.ndarray:
    """Resample a clip ``factor`` times slower (``factor`` > 1 = longer) at the same ``fps_hz``
    (linear in joints / position, nlerp for the root quaternion)."""
    if not factor > 0.0:
        raise ValueError(f"time scale factor must be positive, got {factor}")
    n = qpos.shape[0]
    f = np.clip(np.arange(0.0, (n - 1) * factor + 1e-9, 1.0) / factor, 0.0, n - 1.0)
    i0 = np.clip(np.floor(f).astype(np.int64), 0, n - 2)
    a = (f - i0)[:, None]
    out = qpos[i0] * (1.0 - a) + qpos[i0 + 1] * a
    out[:, 3:7] = quat.quat_nlerp(qpos[i0, 3:7], qpos[i0 + 1, 3:7], a[:, 0], np)
    return out


# ----------------------------------------------------------------------
# Retarget quality metrics (cm / degrees), all on the final G1 qpos.
# ----------------------------------------------------------------------
ERROR_KEYPOINTS: tuple[str, ...] = (
    "lknee", "rknee", "lankle", "rankle", "ltoe", "rtoe", "lsh", "rsh", "lelb", "relb", "lwri", "rwri", "lpalm",
    "rpalm", "headtop",
)  # fmt: skip
LIMIT_TOL_RAD = 1e-6
LIMIT_SATURATION_RAD = 0.01


def keypoint_errors_cm(rig: G1Rig, qpos: np.ndarray, h: HumanMotion) -> dict[str, dict[str, float]]:
    """Mean / max over frames of ``|(p_g1 - hipmid_g1) - (p_human - hipmid_human)|`` per key point (cm).

    ``h`` is the Froude-scaled human motion, so this includes the proportion mismatch between the
    scaled human and the G1 (the IK matches directions, not positions)."""
    err = {k: np.zeros(len(qpos)) for k in ERROR_KEYPOINTS}
    for t, q in enumerate(qpos):
        rig.forward(q)
        hip = rig.point("hipmid")[0]
        for k in ERROR_KEYPOINTS:
            err[k][t] = np.linalg.norm((rig.point(k)[0] - hip) - (h.points[k][t] - h.points["hipmid"][t]))
    return {k: {"mean_cm": 100.0 * float(v.mean()), "max_cm": 100.0 * float(v.max())} for k, v in err.items()}


def joint_limit_report(rig: G1Rig, qpos: np.ndarray) -> dict[str, object]:
    """Frames outside the joint ranges (violations) and frames within ``LIMIT_SATURATION_RAD`` of a limit."""
    j = qpos[:, 7:]
    below, above = rig.lo - j, j - rig.hi
    viol = (below > LIMIT_TOL_RAD) | (above > LIMIT_TOL_RAD)
    sat = (j < rig.lo + LIMIT_SATURATION_RAD) | (j > rig.hi - LIMIT_SATURATION_RAD)
    names = g1_model.G1_JOINT_NAMES
    return {
        "violating_frames": int(np.sum(viol.any(axis=1))),
        "max_violation_rad": float(max(below.max(), above.max(), 0.0)),
        "saturated_frame_fraction_by_joint": {n: float(f) for n, f in zip(names, sat.mean(axis=0)) if f > 0.0},
    }


def floor_clearance_m(rig: G1Rig, q: np.ndarray) -> float:
    """Lowest point of any floor collider (sole corners, capsule/sphere surfaces) above z = 0 (negative = penetration)."""
    rig.forward(q)
    low = min(p[2] for side in "lr" for p, _ in rig.foot_corners(side))
    for gid, radius, ends in rig.floor_ends:
        low = min(low, min(p[2] - radius for p, _ in rig.geom_ends(gid, ends, False)))
    return float(low)


def contact_quality(rig: G1Rig, qpos: np.ndarray, contacts: dict[str, np.ndarray]) -> dict[str, object]:
    """Per contact run: max height error, slip (max xy distance of a locked point from its run mean) in cm and, for
    feet, the sole tilt (deg), plus the global maximum penetration (cm)."""
    runs = []
    for ee, mask in contacts.items():
        side = ee[0]
        for a, b in _runs(mask):
            pts, tilt = [], []
            for k in range(a, b):
                rig.forward(qpos[k])
                if ee.endswith("foot"):
                    pts.append([p for p, _ in rig.foot_corners(side)])
                    rot = rig.body_rot(rig.points[f"{side}ankle"][0])
                    tilt.append(np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0))))
                    floor_z = 0.0
                else:
                    pts.append([p for p, _ in rig.hand_ends(side)])
                    floor_z = HAND_RADIUS_M
            arr = np.array(pts)  # (n, points, 3)
            slip = np.linalg.norm(arr[:, :, :2] - arr[:, :, :2].mean(axis=0), axis=-1).max()
            runs.append({
                "end_effector": ee, "frames": [a, b], "max_height_error_cm": 100.0 * float(np.abs(arr[:, :, 2] - floor_z).max()),
                "max_slip_cm": 100.0 * float(slip),
                "max_sole_tilt_deg": float(max(tilt)) if tilt else None,
            })  # fmt: skip
    clearance = np.array([floor_clearance_m(rig, q) for q in qpos])
    return {
        "runs": runs,
        "max_penetration_cm": 100.0 * float(max(0.0, -clearance.min())),
        "min_clearance_m": float(clearance.min()),
    }


def to_clip(
    rig: G1Rig, qpos: np.ndarray, fps_hz: float, foot_contact: np.ndarray, licence: str, source: str
) -> reference.ReferenceClip:
    """Package a retargeted ``qpos`` as a repo clip: finite-difference ``qvel``, FK site positions."""
    ids = [rig.m.site(s).id for s in g1_model.END_EFFECTOR_SITES]
    sites = []
    for q in qpos:
        rig.forward(q)
        sites.append(rig.d.site_xpos[ids].copy())
    return reference.ReferenceClip(
        fps_hz=fps_hz,
        qpos=qpos,
        qvel=reference.finite_difference_qvel(qpos, fps_hz),
        joint_names=g1_model.G1_JOINT_NAMES,
        site_names=g1_model.END_EFFECTOR_SITES,
        site_pos=np.array(sites),
        licence=licence,
        source=source,
        foot_contact=np.asarray(foot_contact, dtype=bool),
    )
