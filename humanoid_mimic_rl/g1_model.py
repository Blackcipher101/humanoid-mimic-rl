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
masses or inertias. ``SELF_COLLISION_PAIRS`` adds robot self-collision pairs (limbs, trunk, head, hands).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

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

# Robot self-collision pairs (playground's model has none: contype = conaffinity = 0 everywhere).
# Chosen by measured geom distance (mj_geomDistance) at the home pose and over the retargeted CMU
# backflip / spin-kick / cartwheel clips: every pair that came within ~0.2 m in those motions, minus
# pairs that already overlap at the home pose because they meet at a joint (pelvis-thigh at the hip,
# thigh-shin of the same leg at the knee, shin-foot of the same leg at the ankle) and pairs that never
# came closer than 0.2 m (head-foot, pelvis-foot, torso-foot, hand-foot, pelvis-head, pelvis-torso
# across the waist). Same-leg / same-arm chains are excluded for the same reason.
SELF_COLLISION_PAIRS: tuple[tuple[str, str], ...] = (
    # leg vs other leg
    ("left_thigh", "right_thigh"), ("left_shin", "right_shin"),
    ("left_thigh", "right_shin"), ("right_thigh", "left_shin"),
    ("left_foot", "right_foot"), ("left_shin", "right_foot"), ("right_shin", "left_foot"),
    # trunk / head vs legs (tucks, knee to chest)
    ("torso_collision", "left_thigh"), ("torso_collision", "right_thigh"),
    ("torso_collision", "left_shin"), ("torso_collision", "right_shin"),
    ("head_collision", "left_shin"), ("head_collision", "right_shin"),
    # hands vs everything they can reach
    ("left_hand_collision", "right_hand_collision"),
    ("left_hand_collision", "pelvis_collision"), ("right_hand_collision", "pelvis_collision"),
    ("left_hand_collision", "torso_collision"), ("right_hand_collision", "torso_collision"),
    ("left_hand_collision", "head_collision"), ("right_hand_collision", "head_collision"),
    ("left_hand_collision", "left_thigh"), ("left_hand_collision", "right_thigh"),
    ("right_hand_collision", "left_thigh"), ("right_hand_collision", "right_thigh"),
    ("left_hand_collision", "left_shin"), ("left_hand_collision", "right_shin"),
    ("right_hand_collision", "left_shin"), ("right_hand_collision", "right_shin"),
)  # fmt: skip
SELF_COLLISION_FRICTION = 0.5


def home_pose() -> np.ndarray:
    return np.asarray(G1_HOME_JOINT_POS, dtype=np.float64)


def joint_index(name: str) -> int:
    return G1_JOINT_NAMES.index(name)


def make_full_collision_xml(feet_only_xml: str) -> str:
    """Return the G1 robot XML with trunk/head/limb colliders paired with the floor and with each other
    (``SELF_COLLISION_PAIRS``)."""
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
    mu = f"{SELF_COLLISION_FRICTION} {SELF_COLLISION_FRICTION}"
    for g1, g2 in SELF_COLLISION_PAIRS:
        for g in (g1, g2):
            if g not in existing:
                raise ValueError(f"self-collision geom {g!r} missing from G1 XML")
        ET.SubElement(contacts[0], "pair", {"name": f"self_{g1}__{g2}", "geom1": g1, "geom2": g2,
                                            "condim": "3", "friction": mu})
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
# FLIP tasks only (g1_backflip / g1_mimic / g1_supermimic): torque motors limited
# to the real G1 EDU peak torques instead of playground's soft position servos
# (whose reachable torque kp * |ctrl limit - q| is far below the motor limits).
#
# Source: unitree.com/g1 spec table, G1 EDU "Maximum Torque of Knee Joint: 120 N.m"
# (G1 base: 90 N.m), footnote [1]: "The maximum torque of the joint motors of the
# whole machine is different. This is the maximum torque of the largest joint
# motor among them." Unitree publishes no other per-joint figure, so every joint
# keeps the model's force limit (Unitree g1_description effort limits, via
# playground / Menagerie), capped at that 120 N m. Effect: knee and hip roll
# 139 -> 120 N m; all other joints unchanged (hip pitch/yaw 88, ankle 50, waist
# yaw 88, waist roll/pitch 50, shoulder/elbow/wrist roll 25, wrist pitch/yaw 5).
# ----------------------------------------------------------------------
G1_EDU_MAX_JOINT_TORQUE_NM = 120.0


def torque_scale_label(scale: float) -> str:
    """Human-readable disclaimer for any artefact produced with a torque scale != 1."""
    if scale == 1.0:
        return "G1 with published peak torques"
    return (
        f"G1 with {scale:g}x published peak torque - a modified model, not the real robot's limits"
    )

# PD-to-torque layer of the flip envs (a design choice, not Unitree gains):
# full peak torque at FLIP_PD_FULL_TORQUE_ERROR_RAD of position error, damping
# kd = FLIP_PD_KD_PER_KP * kp, targets may exceed the joint range by
# FLIP_PD_TARGET_SLACK_RAD so the full peak torque is reachable at the limits
# (the dynamics checker's cap is then exactly the peak torque).
FLIP_PD_FULL_TORQUE_ERROR_RAD = 0.25
FLIP_PD_KD_PER_KP = 0.02  # s
FLIP_PD_TARGET_SLACK_RAD = 0.25


def flip_peak_torques(model: object, torque_scale: float = 1.0) -> np.ndarray:
    """Per-actuator peak torque (N m) for the flip tasks:
    ``torque_scale * min(model force limit, 120 N m)``.

    ``torque_scale`` > 1 is an explicit owner decision for the flip tasks: the
    published limits did not admit a feasible backflip in the trajectory
    optimiser. Anything produced with it must carry ``torque_scale_label``.
    """
    if not torque_scale > 0.0:
        raise ValueError(f"torque_scale must be positive, got {torque_scale}")
    jnt = model.actuator_trnid[:, 0]
    model_limit = np.max(np.abs(model.jnt_actfrcrange[jnt]), axis=1)
    return torque_scale * np.minimum(model_limit, G1_EDU_MAX_JOINT_TORQUE_NM)


def clip_report_path(clip_path: Path) -> Path:
    """``<clip>.report.json`` written next to a clip by ``scripts/make_g1_backflip.py``."""
    return Path(clip_path).with_suffix(".report.json")


def resolve_torque_scale(clip_path: Path, requested: float | None, allow_infeasible: bool = False) -> float:
    """Torque scale a G1 flip env must use for ``clip_path``.

    The clip's report records the scale the clip was optimised and checked
    with; training on it with a different scale would make the reference
    infeasible (or the robot stronger than the check). ``requested`` (e.g. a
    ``--torque-scale`` flag) must match it if given.

    ``allow_infeasible``: accept a clip whose report says it FAILED the
    feasibility check (e.g. retargeted human mocap that exceeds the G1's
    torques). The robot keeps the recorded torque limits, so the policy can
    only approximate the clip; callers must record this choice (run_config)
    and say so wherever results are reported.
    """
    report = clip_report_path(clip_path)
    if not report.is_file():
        raise ValueError(f"{report} not found; build the clip with scripts/make_g1_backflip.py")
    recorded = json.loads(report.read_text(encoding="utf-8")).get("torque_scale")
    if recorded is None:
        raise ValueError(f"{report} has no torque_scale; regenerate the clip")
    if not json.loads(report.read_text(encoding="utf-8")).get("passed", False):
        if not allow_infeasible:
            raise ValueError(f"{clip_path} FAILED the feasibility check ({report}); refusing to use it "
                             "(pass --allow-infeasible-clip to train an approximation at these torque limits)")
        print(f"WARNING: {clip_path} FAILED the feasibility check ({report}); training anyway at torque scale "
              f"{float(recorded)} (--allow-infeasible-clip): the policy can only approximate this clip", flush=True)
    if requested is not None and float(requested) != float(recorded):
        raise ValueError(f"--torque-scale {requested} != {recorded} recorded for {clip_path}")
    return float(recorded)


def flip_pd_gains(peak_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(kp N m/rad, kd N m s/rad)`` of the flip envs' PD-to-torque layer."""
    kp = np.asarray(peak_nm) / FLIP_PD_FULL_TORQUE_ERROR_RAD
    return kp, FLIP_PD_KD_PER_KP * kp


def to_torque_motors(model: object, peak_nm: np.ndarray) -> None:
    """Replace every actuator of a ``mujoco.MjModel`` by a torque motor limited to ``peak_nm``.

    Motor: gain type fixed with gain 1, no bias, ``ctrlrange = [-peak, peak]``;
    the joints' ``actuatorfrcrange`` is set to the same peak so MuJoCo enforces it.
    """
    import mujoco  # noqa: PLC0415  (keeps this module importable without mujoco)

    peak = np.asarray(peak_nm, dtype=np.float64)
    if peak.shape != (model.nu,) or np.any(peak <= 0):
        raise ValueError(f"need {model.nu} positive peak torques")
    for a in range(model.nu):
        model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[a, :] = 0.0
        model.actuator_gainprm[a, 0] = 1.0
        model.actuator_biasprm[a, :] = 0.0
        model.actuator_ctrllimited[a] = 1
        model.actuator_ctrlrange[a] = (-peak[a], peak[a])
        j = model.actuator_trnid[a, 0]
        model.jnt_actfrclimited[j] = 1
        model.jnt_actfrcrange[j] = (-peak[a], peak[a])
