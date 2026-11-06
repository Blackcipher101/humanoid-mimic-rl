"""Booster T1 model helpers (no JAX dependency).

The MJX-ready T1 model shipped with MuJoCo Playground
(``mujoco_playground/_src/locomotion/t1/xmls/t1_mjx_feetonly.xml``) only lets
the foot spheres touch the floor. Get-up and backflip need the trunk, head
and limbs to collide with the ground, so ``make_full_collision_xml`` derives a
variant in which every primitive collision geom collides with the floor.

Collision filtering (MuJoCo rule: a pair collides if
``(contype_a & conaffinity_b) | (contype_b & conaffinity_a)``):

* floor (playground scene):      contype=1, conaffinity=2
* foot spheres (playground):     contype=2, conaffinity=1  -> floor only
* foot boxes (playground):       contype=4, conaffinity=4  -> left/right foot only
* body geoms (this module):      contype=2, conaffinity=1  -> floor only

so body geoms collide with the floor but not with each other or the feet
(no self-collision, as in the playground locomotion tasks).

Collision cylinders are converted to capsules with the same radius and the
same end-to-end extent (half-length reduced by the radius). Capsule-plane
contacts are cheaper in MJX and produce at most 2 contacts instead of up to 4.
This changes collision geometry only; masses and inertias are given by
explicit ``<inertial>`` elements and are unaffected.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping

import numpy as np

# Joint order of qpos[7:] / actuators in the playground T1 model. Verified
# against the compiled model at environment construction time.
T1_JOINT_NAMES: tuple[str, ...] = (
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)

# "home" keyframe joint angles (rad) of the playground T1 scene.
T1_HOME_JOINT_POS: tuple[float, ...] = (
    0.0, 0.0,
    0.0, -1.4, 0.0, -0.4,
    0.0, 1.4, 0.0, 0.4,
    0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
)  # fmt: skip

# Root (Trunk free joint) height of the "home" keyframe (m).
T1_HOME_ROOT_HEIGHT_M = 0.665

# Joint ranges (rad) of the playground T1 model, same order as T1_JOINT_NAMES.
T1_JOINT_RANGES_RAD: tuple[tuple[float, float], ...] = (
    (-1.57, 1.57), (-0.35, 1.22),
    (-3.31, 1.22), (-1.74, 1.57), (-2.27, 2.27), (-2.44, 0.0),
    (-3.31, 1.22), (-1.57, 1.74), (-2.27, 2.27), (0.0, 2.44),
    (-1.57, 1.57),
    (-1.8, 1.57), (-0.2, 1.57), (-1.0, 1.0), (0.0, 2.34), (-0.87, 0.35), (-0.44, 0.44),
    (-1.8, 1.57), (-1.57, 0.2), (-1.0, 1.0), (0.0, 2.34), (-0.87, 0.35), (-0.44, 0.44),
)  # fmt: skip

# Sites tracked as end effectors by the mimic reward.
T1_END_EFFECTOR_SITES: tuple[str, ...] = ("left_foot", "right_foot", "left_hand", "right_hand")

# Foot contact spheres of the playground T1 model (radius 0.01 m).
T1_FOOT_SPHERE_GEOMS: tuple[str, ...] = tuple(
    f"{side}_foot_{i}" for side in ("left", "right") for i in range(1, 5)
)

# Bodies whose ground contact means a fall for the flip tasks.
T1_FALL_BODIES: tuple[str, ...] = ("Trunk", "H2")

FULL_COLLISION_XML_NAME = "t1_mjx_fullcollision.xml"
FEET_ONLY_XML_NAME = "t1_mjx_feetonly.xml"
FLAT_SCENE_XML_NAME = "scene_mjx_feetonly_flat_terrain.xml"

_BODY_CONTYPE = "2"
_BODY_CONAFFINITY = "1"


def joint_index(name: str) -> int:
    """Index of joint ``name`` in qpos[7:] / ctrl."""
    return T1_JOINT_NAMES.index(name)


def home_pose() -> np.ndarray:
    return np.asarray(T1_HOME_JOINT_POS, dtype=np.float64)


def joint_ranges() -> np.ndarray:
    """``(n_joints, 2)`` array of [lower, upper] limits (rad)."""
    return np.asarray(T1_JOINT_RANGES_RAD, dtype=np.float64)


def make_full_collision_xml(feet_only_xml: str) -> str:
    """Return the T1 robot XML with all primitive collision geoms hitting the floor."""
    root = ET.fromstring(feet_only_xml)

    collision_defaults = [
        d for d in root.iter("default") if d.get("class") == "collision"
    ]
    if len(collision_defaults) != 1:
        raise ValueError(
            f"expected exactly one <default class='collision'>, found {len(collision_defaults)}"
        )
    geom_default = collision_defaults[0].find("geom")
    if geom_default is None:
        raise ValueError("<default class='collision'> has no <geom> element")
    geom_default.set("contype", _BODY_CONTYPE)
    geom_default.set("conaffinity", _BODY_CONAFFINITY)

    n_converted = 0
    for geom in root.iter("geom"):
        if geom.get("class") != "collision" or geom.get("type") != "cylinder":
            continue
        size = [float(s) for s in geom.get("size", "").split()]
        if len(size) != 2:
            raise ValueError(f"cylinder collision geom with unexpected size {size}")
        radius, half_length = size
        geom.set("type", "capsule")
        geom.set("size", f"{radius:g} {max(half_length - radius, 0.0):g}")
        n_converted += 1
    if n_converted == 0:
        raise ValueError("no collision cylinders found; T1 XML layout changed?")

    return ET.tostring(root, encoding="unicode")


def make_scene_xml(scene_xml: str, robot_xml_name: str) -> str:
    """Point the scene's ``<include file=...>`` of the robot at ``robot_xml_name``."""
    root = ET.fromstring(scene_xml)
    includes = [i for i in root.iter("include") if i.get("file") == FEET_ONLY_XML_NAME]
    if len(includes) != 1:
        raise ValueError(
            f"expected one <include file='{FEET_ONLY_XML_NAME}'>, found {len(includes)}"
        )
    includes[0].set("file", robot_xml_name)
    return ET.tostring(root, encoding="unicode")


def full_collision_flat_scene(assets: Mapping[str, bytes]) -> tuple[str, dict[str, bytes]]:
    """Scene XML string and asset dict for the full-collision T1 on flat ground.

    ``assets`` is the playground T1 asset dict (``t1.base.get_assets()``),
    which contains the T1 XML files by basename plus the menagerie meshes.
    The returned dict is a copy with the derived robot XML added; pass both to
    ``mujoco.MjModel.from_xml_string``.
    """
    for name in (FEET_ONLY_XML_NAME, FLAT_SCENE_XML_NAME):
        if name not in assets:
            raise KeyError(f"T1 asset {name!r} missing; is mujoco_playground installed?")
    out = dict(assets)
    out[FULL_COLLISION_XML_NAME] = make_full_collision_xml(
        assets[FEET_ONLY_XML_NAME].decode("utf-8")
    ).encode("utf-8")
    scene = make_scene_xml(assets[FLAT_SCENE_XML_NAME].decode("utf-8"), FULL_COLLISION_XML_NAME)
    return scene, out
