"""CMU Graphics Lab Motion Capture Database (ASF/AMC) parser and forward kinematics (numpy only).

Source: http://mocap.cs.cmu.edu ("This dataset of motions is free for all uses"; see the
``LICENCE`` string below for the full statement). Files: ``subjects/<SS>/<SS>.asf`` (skeleton)
and ``subjects/<SS>/<SS>_<TT>.amc`` (motion).

Conventions (verified against the CMU FAQ and the files):

* lengths: ASF bone lengths and AMC root translations are in "ASF units"; the CMU FAQ gives
  ``(1.0 / 0.45) * 2.54 / 100 = 0.056444`` m per unit (0.45 is the ``:units length`` scale in the
  ASF header, 2.54 / 100 converts inches to metres). ``length_to_m`` is derived from the header.
* angles are degrees (``:DEGREES``); a rotation with axis order ``XYZ`` is ``R = Rz Ry Rx``
  (rotations about the fixed axes, x first), for the root, the bone ``axis`` frames and the dofs.
* the world frame of the files is Y-up (the capture volume's up axis). ``to_z_up`` rotates +90 deg
  about x: ``(x, y, z) -> (x, -z, y)`` (right-handed, det = +1), giving the repo's z-up world.
* forward kinematics (Lander, "Working with motion capture file formats"): with ``C_i`` the bone
  ``axis`` rotation and ``M_i`` the rotation from its dofs, the global rotation applied to bone
  ``i`` is ``T_i = T_parent C_i M_i C_i^-1`` (``T_root`` = root rotation); a bone starts at the tip of
  its parent bone (root children start at the root translation) and ends at
  ``origin + T_i * direction * length``.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np

LICENCE = (
    "Motion data from the CMU Graphics Lab Motion Capture Database, http://mocap.cs.cmu.edu: "
    "\"This dataset of motions is free for all uses.\" \"This data is free for use in research projects. "
    "You may include this data in commercially-sold products, but you may not resell this data directly, "
    "even in converted form.\" Acknowledgement requested: \"The data used in this project was obtained from "
    "mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.\" Retargeted to the Unitree "
    "G1 by humanoid_mimic_rl.g1_retarget (this repository's licence applies to the code only); do not "
    "redistribute the converted clips as a dataset."
)
BASE_URL = "http://mocap.cs.cmu.edu/subjects"
_AXES = {"x": 0, "y": 1, "z": 2}


@dataclasses.dataclass(frozen=True)
class Bone:
    name: str
    parent: str  # "root" for the children of the root
    direction: np.ndarray  # (3,) unit vector in the global rest frame (Y-up)
    length: float  # ASF units
    axis_deg: np.ndarray  # (3,) rest-frame rotation of the bone's local axes
    axis_order: str  # e.g. "XYZ"
    dof: tuple[str, ...]  # e.g. ("rx", "ry", "rz")


@dataclasses.dataclass(frozen=True)
class Skeleton:
    bones: tuple[Bone, ...]  # parents before children
    length_to_m: float  # metres per ASF/AMC length unit
    root_order: tuple[str, ...]  # e.g. ("tx", "ty", "tz", "rx", "ry", "rz")
    root_axis_order: str

    def bone_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bones)


@dataclasses.dataclass(frozen=True)
class Motion:
    """AMC frames: ``root`` is (T, 6) = TX TY TZ (ASF units) RX RY RZ (deg), ``angles_deg`` maps bone -> (T, n_dof)."""

    root: np.ndarray
    angles_deg: dict[str, np.ndarray]

    @property
    def num_frames(self) -> int:
        return int(self.root.shape[0])


def rot_xyz(angles_rad: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """Rotation matrices from fixed-axis angles ``(..., 3)`` (about x, y, z); ``order`` "XYZ" -> ``Rz Ry Rx``."""
    a = np.asarray(angles_rad, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    one, zero = np.ones_like(a[..., 0]), np.zeros_like(a[..., 0])
    rx = np.stack([one, zero, zero, zero, c[..., 0], -s[..., 0], zero, s[..., 0], c[..., 0]], -1).reshape(a.shape[:-1] + (3, 3))
    ry = np.stack([c[..., 1], zero, s[..., 1], zero, one, zero, -s[..., 1], zero, c[..., 1]], -1).reshape(a.shape[:-1] + (3, 3))
    rz = np.stack([c[..., 2], -s[..., 2], zero, s[..., 2], c[..., 2], zero, zero, zero, one], -1).reshape(a.shape[:-1] + (3, 3))
    mats = {"x": rx, "y": ry, "z": rz}
    out = np.broadcast_to(np.eye(3), rx.shape).copy()
    for ch in order.lower():  # first listed axis is applied first (rightmost factor)
        out = mats[ch] @ out
    return out


def parse_asf(path: Path) -> Skeleton:
    """Parse a CMU ``.asf`` skeleton (units, root, bonedata, hierarchy)."""
    text = Path(path).read_text(encoding="ascii", errors="replace").replace("(", " ").replace(")", " ")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    section = ""
    length_scale = None
    root_order: tuple[str, ...] = ()
    root_axis = "XYZ"
    raw: dict[str, dict] = {}
    cur: dict | None = None
    hierarchy: list[tuple[str, list[str]]] = []
    for ln in lines:
        if ln.startswith(":"):
            section = ln[1:].split()[0].lower()
            continue
        tok = ln.split()
        if section == "units" and tok[0] == "length":
            length_scale = float(tok[1])
        elif section == "root":
            if tok[0] == "order":
                root_order = tuple(t.lower() for t in tok[1:])
            elif tok[0] == "axis":
                root_axis = tok[1].upper()
        elif section == "bonedata":
            if tok[0] == "begin":
                cur = {"dof": ()}
            elif tok[0] == "end":
                raw[cur["name"]] = cur
                cur = None
            elif cur is not None:
                key = tok[0]
                if key == "name":
                    cur["name"] = tok[1]
                elif key == "direction":
                    cur["direction"] = np.array(tok[1:4], dtype=np.float64)
                elif key == "length":
                    cur["length"] = float(tok[1])
                elif key == "axis":
                    cur["axis"] = np.array(tok[1:4], dtype=np.float64)
                    cur["axis_order"] = tok[4].upper()
                elif key == "dof":
                    cur["dof"] = tuple(t.lower() for t in tok[1:])
        elif section == "hierarchy":
            if tok[0] not in ("begin", "end"):
                hierarchy.append((tok[0], tok[1:]))
    if length_scale is None or not root_order:
        raise ValueError(f"{path}: missing :units length or :root order")
    parent_of = {child: parent for parent, children in hierarchy for child in children}
    ordered: list[Bone] = []
    pending = set(raw)
    known = {"root"}
    while pending:
        progressed = False
        for name in sorted(pending, key=lambda n: list(raw).index(n)):
            if parent_of.get(name) in known:
                b = raw[name]
                ordered.append(
                    Bone(name, parent_of[name], b["direction"], b["length"], b["axis"], b["axis_order"], b["dof"])
                )
                known.add(name)
                pending.discard(name)
                progressed = True
        if not progressed:
            raise ValueError(f"{path}: bones without a parent in :hierarchy: {sorted(pending)}")
    return Skeleton(tuple(ordered), (1.0 / length_scale) * 2.54 / 100.0, root_order, root_axis)


def parse_amc(path: Path, skeleton: Skeleton) -> Motion:
    """Parse a CMU ``.amc`` motion (``:FULLY-SPECIFIED`` frames: a frame number, then ``bone v1 v2 ...``)."""
    dof = {b.name: b.dof for b in skeleton.bones}
    root_rows: list[list[float]] = []
    per_bone: dict[str, list[list[float]]] = {b.name: [] for b in skeleton.bones if b.dof}
    seen: set[str] = set()
    started = False
    for raw in Path(path).read_text(encoding="ascii", errors="replace").splitlines():
        ln = raw.strip()
        if not ln or ln[0] in "#:":
            continue
        tok = ln.split()
        if len(tok) == 1 and tok[0].isdigit():
            if started and seen != set(per_bone) | {"root"}:
                raise ValueError(f"{path}: frame {tok[0]} follows an incomplete frame (not :FULLY-SPECIFIED)")
            started, seen = True, set()
            continue
        name, vals = tok[0], [float(t) for t in tok[1:]]
        if name == "root":
            if len(vals) != len(skeleton.root_order):
                raise ValueError(f"{path}: root has {len(vals)} values, expected {len(skeleton.root_order)}")
            root_rows.append(vals)
        elif name in per_bone:
            if len(vals) != len(dof[name]):
                raise ValueError(f"{path}: bone {name} has {len(vals)} values, expected {len(dof[name])}")
            per_bone[name].append(vals)
        else:
            raise ValueError(f"{path}: unknown bone {name!r}")
        seen.add(name)
    if not root_rows:
        raise ValueError(f"{path}: no frames")
    n = len(root_rows)
    if any(len(v) != n for v in per_bone.values()):
        raise ValueError(f"{path}: bones have differing frame counts")
    idx = {k: i for i, k in enumerate(skeleton.root_order)}
    root = np.array(root_rows)[:, [idx[k] for k in ("tx", "ty", "tz", "rx", "ry", "rz")]]
    return Motion(root, {k: np.array(v) for k, v in per_bone.items()})


def to_z_up(points_y_up: np.ndarray) -> np.ndarray:
    """Y-up -> Z-up (rotation +90 deg about x): ``(x, y, z) -> (x, -z, y)``; works on ``(..., 3)``."""
    p = np.asarray(points_y_up)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pose:
    """Per-bone global ``origin`` / ``tip`` (T, 3) in metres, Y-up, and bone rotation ``rot`` (T, 3, 3)."""

    origin: dict[str, np.ndarray]
    tip: dict[str, np.ndarray]
    rot: dict[str, np.ndarray]
    root_pos: np.ndarray  # (T, 3) m


def forward_kinematics(skeleton: Skeleton, motion: Motion) -> Pose:
    """Global bone origins/tips (m, Y-up) for every frame."""
    t = motion.num_frames
    scale = skeleton.length_to_m
    root_pos = motion.root[:, :3] * scale
    t_root = rot_xyz(np.deg2rad(motion.root[:, 3:6]), skeleton.root_axis_order)
    origin: dict[str, np.ndarray] = {}
    tip: dict[str, np.ndarray] = {}
    rot: dict[str, np.ndarray] = {"root": t_root}
    tip["root"] = root_pos
    for b in skeleton.bones:
        c = rot_xyz(np.deg2rad(b.axis_deg), b.axis_order)
        ang = np.zeros((t, 3))
        if b.dof:
            vals = np.deg2rad(motion.angles_deg[b.name])
            for k, d in enumerate(b.dof):
                ang[:, _AXES[d[1]]] = vals[:, k]
        m = rot_xyz(ang, b.axis_order)
        t_parent = rot[b.parent]
        t_i = t_parent @ (c @ m @ c.T)
        rot[b.name] = t_i
        origin[b.name] = tip[b.parent]
        tip[b.name] = origin[b.name] + np.einsum("tij,j->ti", t_i, b.direction) * (b.length * scale)
    return Pose(origin, tip, rot, root_pos)


@dataclasses.dataclass(frozen=True)
class Trial:
    subject: int
    trial: int
    description: str  # as listed on the CMU subject page
    fps_hz: float  # "framerate" column of the subject page

    @property
    def stem(self) -> str:
        return f"{self.subject:02d}_{self.trial:02d}"

    @property
    def asf_url(self) -> str:
        return f"{BASE_URL}/{self.subject:02d}/{self.subject:02d}.asf"

    @property
    def amc_url(self) -> str:
        return f"{BASE_URL}/{self.subject:02d}/{self.stem}.amc"


# Trials of the CMU acrobatics subjects used here (descriptions / frame rates from
# http://mocap.cs.cmu.edu/search.php?subjectnumber=<SS> as of 2026-11-06).
TRIALS: dict[str, Trial] = {
    t.stem: t
    for t in (
        Trial(90, 2, "cartwheel", 120.0), Trial(90, 3, "cartwheel", 120.0), Trial(90, 4, "cartwheel", 120.0),
        Trial(88, 6, "jump and spin kick", 60.0), Trial(90, 8, "side flip", 120.0), Trial(88, 7, "cartwheel", 60.0),
        Trial(49, 6, "cartwheel", 120.0), Trial(87, 3, "Backflip", 60.0),
    )
}  # fmt: skip
