"""Unitree G1 (29 DoF) model helpers (no JAX dependency).

Source model: MuJoCo Playground 0.2.0 ``locomotion/g1/xmls/g1_mjx_feetonly.xml``
(Apache-2.0), whose meshes come from MuJoCo Menagerie ``unitree_g1`` at the
commit pinned by playground (BSD-3-Clause, Copyright Unitree Robotics).

That model only lets the two foot boxes touch the floor (explicit ``<pair>``
contacts; every geom has ``contype = conaffinity = 0``) and has no collision
geometry on the pelvis, torso or head. ``make_full_collision_xml`` adds

* pelvis sphere, torso capsule and head sphere with the sizes/positions of
  Menagerie's ``g1_mjx.xml`` colliders (same BSD-3 model family), and
* ``<pair>`` contacts with the floor for those plus the existing thigh, shin
  and hand capsules (condim 3, friction 0.6 like playground's foot pairs).

Every G1 body carries an explicit ``<inertial>``, so added geoms do not change
masses or inertias. No new self-collision pairs are added.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping

import numpy as np

ROBOT_XML_NAME = "g1_mjx_feetonly.xml"
FULL_COLLISION_XML_NAME = "g1_mjx_fullcollision.xml"
FLAT_SCENE_XML_NAME = "scene_mjx_feetonly_flat_terrain.xml"

G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)  # fmt: skip

# "home" keyframe of the playground scene (rad) and its pelvis height (m).
G1_HOME_JOINT_POS: tuple[float, ...] = (
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 1.28, 0.0, 0.0, 0.0,
)  # fmt: skip
G1_HOME_ROOT_HEIGHT_M = 0.785

ROOT_BODY = "pelvis"
FOOT_GEOMS: tuple[str, ...] = ("left_foot", "right_foot")
FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
# Foot box (playground): centre (0.04, 0, -0.029) in the ankle-roll frame,
# half-sizes (0.09, 0.03, 0.008) m. Sole centre and half-extents:
FOOT_SOLE_CENTRE_LOCAL_M: tuple[float, float, float] = (0.04, 0.0, -0.037)
FOOT_HALF_LENGTH_M = 0.09
FOOT_HALF_WIDTH_M = 0.03
FOOT_FRICTION = 0.6
END_EFFECTOR_SITES: tuple[str, ...] = ("left_foot", "right_foot", "left_palm", "right_palm")
FALL_GEOMS: tuple[str, ...] = ("pelvis_collision", "torso_collision", "head_collision")
FOOT_CONTACT_SENSORS: tuple[str, ...] = ("left_foot_floor_found", "right_foot_floor_found")

# Colliders added to the playground model: (body, name, type, size, placement attr, value).
_ADDED_GEOMS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("pelvis", "pelvis_collision", "sphere", "0.07", "pos", "0 0 -0.08"),
    ("torso_link", "torso_collision", "capsule", "0.09", "fromto", "0.01 0 0.08 0.01 0 0.2"),
    ("torso_link", "head_collision", "sphere", "0.06", "pos", "0 0 0.43"),
)
FLOOR_PAIR_GEOMS: tuple[str, ...] = (
    "pelvis_collision", "torso_collision", "head_collision",
    "left_thigh", "right_thigh", "left_shin", "right_shin",
    "left_hand_collision", "right_hand_collision",
)  # fmt: skip


def home_pose() -> np.ndarray:
    return np.asarray(G1_HOME_JOINT_POS, dtype=np.float64)


def joint_index(name: str) -> int:
    return G1_JOINT_NAMES.index(name)


def make_full_collision_xml(feet_only_xml: str) -> str:
    """Return the G1 robot XML with trunk/head/limb colliders paired with the floor."""
    root = ET.fromstring(feet_only_xml)
    bodies = {b.get("name"): b for b in root.iter("body")}
    existing = {g.get("name") for g in root.iter("geom") if g.get("name")}
    for body, name, gtype, size, attr, value in _ADDED_GEOMS:
        if body not in bodies:
            raise ValueError(f"G1 XML has no body {body!r}")
        if name in existing:
            raise ValueError(f"G1 XML already has a geom named {name!r}")
        ET.SubElement(
            bodies[body], "geom", {"name": name, "class": "collision", "type": gtype, "size": size, attr: value}
        )
        existing.add(name)
    contacts = root.findall("contact")
    if len(contacts) != 1:
        raise ValueError(f"expected one <contact> section, found {len(contacts)}")
    for g in FLOOR_PAIR_GEOMS:
        if g not in existing:
            raise ValueError(f"collision geom {g!r} missing from G1 XML")
        ET.SubElement(
            contacts[0], "pair",
            {"name": f"{g}_floor", "geom1": g, "geom2": "floor", "condim": "3", "friction": "0.6 0.6"},
        )
    return ET.tostring(root, encoding="unicode")


def make_scene_xml(scene_xml: str, robot_xml_name: str) -> str:
    root = ET.fromstring(scene_xml)
    includes = [i for i in root.iter("include") if i.get("file") == ROBOT_XML_NAME]
    if len(includes) != 1:
        raise ValueError(f"expected one <include file='{ROBOT_XML_NAME}'>, found {len(includes)}")
    includes[0].set("file", robot_xml_name)
    return ET.tostring(root, encoding="unicode")


def full_collision_flat_scene(assets: Mapping[str, bytes]) -> tuple[str, dict[str, bytes]]:
    """Scene XML and asset dict (copy) for the full-collision G1 on flat ground.

    ``assets`` is playground's ``g1.base.get_assets()`` dict (XMLs + meshes by basename).
    """
    for name in (ROBOT_XML_NAME, FLAT_SCENE_XML_NAME):
        if name not in assets:
            raise KeyError(f"G1 asset {name!r} missing; is mujoco_playground installed?")
    out = dict(assets)
    out[FULL_COLLISION_XML_NAME] = make_full_collision_xml(assets[ROBOT_XML_NAME].decode("utf-8")).encode("utf-8")
    return make_scene_xml(assets[FLAT_SCENE_XML_NAME].decode("utf-8"), FULL_COLLISION_XML_NAME), out


# ----------------------------------------------------------------------
# Servo stiffness for the FLIP tasks only (g1_backflip / g1_mimic / g1_supermimic).
# Playground's G1 uses kp 20 (ankle pitch) and kp 2 (ankle roll); with position
# servos the reachable torque is kp * |ctrl limit - q|, so a plantar-flexed ankle
# can only push with a few N m although its force limit is 50 N m.
# Values from Unitree's public G1 gains (both BSD-3-Clause, Unitree Robotics):
#   * unitree_rl_gym @ 276801e, legged_gym/envs/g1/g1_config.py: stiffness 'ankle': 40
#   * unitree_sdk2 @ 8e1b927, example/g1/low_level/g1_ankle_swing_example.cpp:
#     Kp ankle pitch/roll 40, waist yaw 60, waist roll/pitch 40
# Waist: playground's kp 75 is already above Unitree's public 40, so it is kept.
# (Menagerie's non-MJX g1.xml uses kp 500 everywhere but marks the actuators
# "needs tuning", so it is not used as evidence.) Force limits are unchanged.
# ----------------------------------------------------------------------
FLIP_SERVO_KP: dict[str, float] = {
    "left_ankle_pitch_joint": 40.0,
    "right_ankle_pitch_joint": 40.0,
    "left_ankle_roll_joint": 40.0,
    "right_ankle_roll_joint": 40.0,
}


def apply_servo_kp(model: object, kp_by_joint: Mapping[str, float]) -> None:
    """Set position-servo stiffness of ``mujoco.MjModel`` actuators driving the named joints.

    MuJoCo position actuators compute ``kp * ctrl - kp * q - kv * qd``: gain
    ``gainprm[0] = kp`` and bias ``biasprm[1] = -kp``; both are updated so the
    servo stays consistent. Raises ``KeyError`` for unknown joints.
    """
    joint_of_actuator = {model.joint(int(model.actuator_trnid[a, 0])).name: a for a in range(model.nu)}
    for name, kp in kp_by_joint.items():
        a = joint_of_actuator[name]
        model.actuator_gainprm[a, 0] = kp
        model.actuator_biasprm[a, 1] = -kp
