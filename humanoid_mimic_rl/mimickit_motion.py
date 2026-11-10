"""MimicKit humanoid motion clips (``data/motions/humanoid/*.pkl``) -> human key points for ``g1_retarget``.

MimicKit (https://github.com/xbpeng/MimicKit, code Apache-2.0) ships its motion data as a separate download
whose licence is not stated; clips derived from it are for local training only and are not redistributed.

Motion file: a pickle of ``{"loop_mode": int, "fps": int, "frames": (T, 6 + D)}``. A frame is the root position
(m), the root rotation as an exponential map (rad), then the joint DoFs in MJCF body order: a body with three
hinge joints is one spherical joint stored as an exponential map (3 values), a body with one hinge is its angle
(``mimickit/anim/kin_char_model.py``, ``mjcf_char_model.py``). Body rotation = parent * exp(dof); the humanoid
MJCF has identity body frames, so MuJoCo's x-y-z hinge chain reproduces it with intrinsic XYZ Euler angles.

The pickle is read with a restricted unpickler (numpy arrays and builtin containers only), so a tampered file
cannot run code.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from humanoid_mimic_rl import g1_retarget

LICENCE = (
    "Motion data from MimicKit (https://github.com/xbpeng/MimicKit; code Apache-2.0, motion data download with no "
    "stated licence). Retargeted to the Unitree G1 by humanoid_mimic_rl.g1_retarget; local training use only, do "
    "not redistribute the converted clip."
)
KEYPOINT_RATE_HZ = 120.0  # key points are spline-resampled to this rate before retargeting (clips are 16-60 fps)

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("builtins", "dict"), ("builtins", "list"),
}  # fmt: skip


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: ANN201
        if (module, name) in _ALLOWED_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"MimicKit motion file references a disallowed global {module}.{name}")


def load_motion(path: Path) -> tuple[float, np.ndarray]:
    """``(fps_hz, frames (T, 6 + D))`` from a MimicKit motion pickle."""
    with path.open("rb") as f:
        data = _RestrictedUnpickler(f).load()
    if not isinstance(data, dict) or "fps" not in data or "frames" not in data:
        raise ValueError(f"{path}: not a MimicKit motion file (keys {sorted(data) if isinstance(data, dict) else type(data)})")
    frames = np.asarray(data["frames"], dtype=float)
    fps = float(data["fps"])
    if frames.ndim != 2 or frames.shape[0] < 2 or frames.shape[1] < 6 or not np.all(np.isfinite(frames)) or fps <= 0:
        raise ValueError(f"{path}: invalid frames {frames.shape} / fps {fps}")
    return fps, frames


class HumanoidFK:
    """Forward kinematics of a MimicKit humanoid MJCF for its motion frames."""

    def __init__(self, xml_path: Path) -> None:
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        if self.m.njnt < 1 or self.m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"{xml_path}: first joint must be the free root joint")
        # DoF groups in body order: (qpos addresses of the body's hinges), one group per body with joints.
        self.groups: list[list[int]] = []
        for b in range(1, self.m.nbody):
            jnts = [j for j in range(1, self.m.njnt) if self.m.jnt_bodyid[j] == b]  # joint 0: free root
            if not jnts:
                continue
            if any(self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in jnts) or len(jnts) not in (1, 3):
                raise ValueError(f"body {self.body_name(b)}: only 1 or 3 hinge joints are supported")
            if len(jnts) == 3:
                axes = np.array([self.m.jnt_axis[j] for j in jnts])
                if not np.allclose(axes, np.eye(3)):
                    raise ValueError(f"body {self.body_name(b)}: spherical joint hinges must be x, y, z in order")
            self.groups.append([int(self.m.jnt_qposadr[j]) for j in jnts])
        self.dof_dim = sum(len(g) for g in self.groups)

    def body_name(self, b: int) -> str:
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)

    def qpos(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] != 6 + self.dof_dim:
            raise ValueError(f"frame has {frame.shape[0]} values, model expects 6 + {self.dof_dim}")
        q = np.zeros(self.m.nq)
        q[:3] = frame[:3]
        x, y, z, w = Rotation.from_rotvec(frame[3:6]).as_quat()
        q[3:7] = (w, x, y, z)
        i = 6
        for g in self.groups:
            if len(g) == 3:
                q[g] = Rotation.from_rotvec(frame[i : i + 3]).as_euler("XYZ")
                i += 3
            else:
                q[g[0]] = frame[i]
                i += 1
        return q

    def point(self, body: str, local: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        bid = self.m.body(body).id
        return self.d.xpos[bid] + self.d.xmat[bid].reshape(3, 3) @ np.asarray(local, dtype=float)


# Key point -> (body, offset in the body frame, m) for the MimicKit/DeepMimic humanoid (humanoid.xml).
HUMANOID_KEYPOINTS: dict[str, tuple[str, tuple[float, float, float]]] = {
    "lhip": ("left_thigh", (0, 0, 0)), "rhip": ("right_thigh", (0, 0, 0)),
    "lknee": ("left_shin", (0, 0, 0)), "rknee": ("right_shin", (0, 0, 0)),
    "lankle": ("left_foot", (0, 0, 0)), "rankle": ("right_foot", (0, 0, 0)),
    "ltoe": ("left_foot", (0.1335, 0, -0.025)), "rtoe": ("right_foot", (0.1335, 0, -0.025)),  # box front edge
    "lsh": ("left_upper_arm", (0, 0, 0)), "rsh": ("right_upper_arm", (0, 0, 0)),
    "lelb": ("left_lower_arm", (0, 0, 0)), "relb": ("right_lower_arm", (0, 0, 0)),
    "lwri": ("left_hand", (0, 0, 0)), "rwri": ("right_hand", (0, 0, 0)),
    "lpalm": ("left_hand", (0, 0.04, 0)), "rpalm": ("right_hand", (0, -0.04, 0)),  # hand sphere tip along the arm
    "headtop": ("head", (0, 0, 0.27)), "spine_low": ("torso", (0, 0, 0)), "neck_base": ("head", (0, 0, 0)),
}  # fmt: skip


def human_keypoints(fk: HumanoidFK, fps_hz: float, frames: np.ndarray,
                    out_rate_hz: float = KEYPOINT_RATE_HZ) -> g1_retarget.HumanMotion:  # fmt: skip
    """Key points of a MimicKit humanoid motion (z-up already), spline-resampled to ``out_rate_hz``."""
    raw: dict[str, list[np.ndarray]] = {k: [] for k in HUMANOID_KEYPOINTS}
    for frame in frames:
        fk.d.qpos[:] = fk.qpos(frame)
        mujoco.mj_kinematics(fk.m, fk.d)
        for name, (body, local) in HUMANOID_KEYPOINTS.items():
            raw[name].append(fk.point(body, local))
    t = np.arange(frames.shape[0]) / fps_hz
    t_out = np.arange(0.0, t[-1] + 1e-9, 1.0 / out_rate_hz)
    pts = {k: CubicSpline(t, np.array(v), axis=0)(t_out) for k, v in raw.items()}
    return g1_retarget.keypoints_to_motion(pts, out_rate_hz)
