"""CMU ASF/AMC parser, G1 retargeting output schema and the contact-aware feasibility report.

The parser unit tests run on a tiny synthetic skeleton. Tests on real data need the downloaded CMU files
(``CMU_MOCAP_DIR`` with ``90.asf``, ``90_02.amc``, ``88.asf``, ``88_06.amc``) and the G1 model files
(``G1_XML_DIR`` playground ``locomotion/g1/xmls``, ``G1_MESH_DIR`` menagerie ``unitree_g1/assets``); they skip otherwise.
"""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_mimic_rl import cmu_mocap, g1_model, reference

_ASF = """:version 1.10
:name test
:units
  mass 1.0
  length 0.45
  angle deg
:root
   order TX TY TZ RX RY RZ
   axis XYZ
   position 0 0 0
   orientation 0 0 0
:bonedata
  begin
     id 1
     name spine
     direction 0 1 0
     length 10.0
     axis 0 0 0  XYZ
     dof rz
     limits (-90.0 90.0)
  end
  begin
     id 2
     name arm
     direction 1 0 0
     length 5.0
     axis 0 0 0  XYZ
  end
:hierarchy
  begin
    root spine
    spine arm
  end
"""
_AMC = """#!OML:ASF test
:FULLY-SPECIFIED
:DEGREES
1
root 100 0 0 0 0 0
spine 0
2
root 100 0 0 0 0 90
spine 90
"""

_CMU = os.environ.get("CMU_MOCAP_DIR")
_XML, _MESH = os.environ.get("G1_XML_DIR"), os.environ.get("G1_MESH_DIR")
needs_cmu = pytest.mark.skipif(not _CMU, reason="CMU_MOCAP_DIR not set")
needs_g1 = pytest.mark.skipif(not (_XML and _MESH), reason="G1_XML_DIR / G1_MESH_DIR not set")


def _assets() -> dict[str, bytes]:
    xml = Path(_XML)
    files = [*xml.glob("*.xml"), *(xml / "assets").glob("*"), *Path(_MESH).glob("*")]
    return {f.name: f.read_bytes() for f in files if f.is_file()}


@pytest.fixture()
def tiny(tmp_path):
    (tmp_path / "t.asf").write_text(_ASF)
    (tmp_path / "t.amc").write_text(_AMC)
    skeleton = cmu_mocap.parse_asf(tmp_path / "t.asf")
    return skeleton, cmu_mocap.parse_amc(tmp_path / "t.amc", skeleton)


def test_length_unit_matches_cmu_faq(tiny):
    skeleton, _ = tiny
    assert skeleton.length_to_m == pytest.approx((1.0 / 0.45) * 2.54 / 100.0)
    assert skeleton.length_to_m == pytest.approx(0.056444, abs=1e-6)


def test_parse_counts_and_order(tiny):
    skeleton, motion = tiny
    assert skeleton.bone_names() == ("spine", "arm")
    assert skeleton.bones[0].dof == ("rz",) and skeleton.bones[1].dof == ()
    assert skeleton.root_order == ("tx", "ty", "tz", "rx", "ry", "rz")
    assert motion.num_frames == 2
    np.testing.assert_allclose(motion.root[1], [100, 0, 0, 0, 0, 90])
    np.testing.assert_allclose(motion.angles_deg["spine"], [[0.0], [90.0]])


def test_forward_kinematics_units_and_axes(tiny):
    skeleton, motion = tiny
    pose = cmu_mocap.forward_kinematics(skeleton, motion)
    s = skeleton.length_to_m
    # frame 0: root at (100 units, 0, 0) m, spine straight up (+y in the Y-up file frame), arm along +x
    np.testing.assert_allclose(pose.root_pos[0], [100 * s, 0, 0], atol=1e-9)
    np.testing.assert_allclose(pose.tip["spine"][0], [100 * s, 10 * s, 0], atol=1e-9)
    np.testing.assert_allclose(pose.tip["arm"][0], [100 * s + 5 * s, 10 * s, 0], atol=1e-9)
    # frame 1: root rz = 90 deg and spine rz = 90 deg -> total 180 deg about z: the spine points down
    np.testing.assert_allclose(pose.tip["spine"][1], [100 * s, -10 * s, 0], atol=1e-9)
    np.testing.assert_allclose(pose.tip["arm"][1], [100 * s - 5 * s, -10 * s, 0], atol=1e-9)
    z_up = cmu_mocap.to_z_up(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(z_up, [[1.0, -3.0, 2.0]])  # Y-up -> Z-up, right-handed
    rot = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    assert np.linalg.det(rot) == pytest.approx(1.0)
    np.testing.assert_allclose(rot @ np.array([1.0, 2.0, 3.0]), z_up[0])


def test_amc_rejects_bad_dof_count(tmp_path, tiny):
    skeleton, _ = tiny
    (tmp_path / "bad.amc").write_text(":FULLY-SPECIFIED\n:DEGREES\n1\nroot 0 0 0 0 0 0\nspine 0 0\n")
    with pytest.raises(ValueError, match="spine"):
        cmu_mocap.parse_amc(tmp_path / "bad.amc", skeleton)


def test_trial_table_urls():
    t = cmu_mocap.TRIALS["90_02"]
    assert (t.description, t.fps_hz) == ("cartwheel", 120.0)
    assert t.amc_url == "http://mocap.cs.cmu.edu/subjects/90/90_02.amc"
    assert t.asf_url == "http://mocap.cs.cmu.edu/subjects/90/90.asf"
    assert cmu_mocap.TRIALS["88_06"].fps_hz == 60.0
    assert "free for all uses" in cmu_mocap.LICENCE


@needs_cmu
def test_real_files_frame_and_bone_counts():
    raw = Path(_CMU)
    sk = cmu_mocap.parse_asf(raw / "90.asf")
    assert len(sk.bones) == 30 and sum(len(b.dof) for b in sk.bones) == 56
    assert cmu_mocap.parse_amc(raw / "90_02.amc", sk).num_frames == 781
    sk88 = cmu_mocap.parse_asf(raw / "88.asf")
    assert cmu_mocap.parse_amc(raw / "88_06.amc", sk88).num_frames == 230


@needs_cmu
def test_real_skeleton_leg_length_in_metres():
    raw = Path(_CMU)
    sk = cmu_mocap.parse_asf(raw / "90.asf")
    motion = cmu_mocap.parse_amc(raw / "90_02.amc", sk)
    pose = cmu_mocap.forward_kinematics(sk, motion)
    femur = np.linalg.norm(pose.tip["lfemur"][0] - pose.origin["lfemur"][0])
    assert femur == pytest.approx(7.24654 * 0.056444, rel=1e-4)  # 0.409 m: a human thigh, not inches
    head_z = cmu_mocap.to_z_up(pose.tip["head"])[0, 2]
    assert 1.4 < head_z < 1.9  # standing person, z up


def test_froude_scaling_and_masks():
    from humanoid_mimic_rl import g1_retarget as rt

    t = np.linspace(0.0, 2.0, 101)  # 50 Hz, 2 s
    p = {k: np.stack([t, 0 * t, 1.0 + 0 * t], axis=1) for k in rt.KEYPOINT_NAMES}
    eye = np.tile(np.eye(3), (len(t), 1, 1))
    h = rt.HumanMotion(50.0, p, eye, eye, 0.9)
    out = rt.froude_resample(h, 0.64, 25.0)  # sqrt(0.64) = 0.8
    assert out.num_frames == int(2.0 * 0.8 * 25) + 1
    assert out.leg_length_m == pytest.approx(0.9 * 0.64)
    np.testing.assert_allclose(out.points["lhip"][-1], [0.64 * 2.0, 0.0, 0.64], atol=1e-9)  # lengths scale by s
    mask = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0], dtype=bool)
    cleaned = rt.clean_mask(mask, min_frames=3, max_gap=3)
    assert cleaned.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]


def test_time_scale_resampling():
    from humanoid_mimic_rl import g1_retarget as rt

    q = np.zeros((11, 9))
    q[:, 3] = 1.0
    q[:, 0] = np.arange(11) * 0.1
    slow = rt.apply_time_scale(q, 50.0, 2.0)
    assert slow.shape[0] == 21 and slow[-1, 0] == pytest.approx(1.0) and slow[1, 0] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        rt.apply_time_scale(q, 50.0, 0.0)


# ----------------------------------------------------------------------
# G1 model dependent tests.
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def rig():
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    from humanoid_mimic_rl import g1_retarget

    return g1_retarget.G1Rig(_assets())


@pytest.fixture(scope="module")
def fb(rig):
    from humanoid_mimic_rl import g1_backflip

    return g1_backflip.floating_base(rig.m)


def _home_clip(rig, n=30):
    q = rig.home_q.copy()
    q[2] = 0.785 - rig_min_clearance(rig, q)
    return np.tile(q, (n, 1))


def rig_min_clearance(rig, q):
    from humanoid_mimic_rl import g1_retarget

    return g1_retarget.floor_clearance_m(rig, q)


@needs_g1
def test_static_stand_is_feasible_with_hands_and_feet_logic(rig, fb):
    from humanoid_mimic_rl import g1_cmu_feasibility as gf

    rep = gf.check_clip(fb, _home_clip(rig), 50.0)
    assert rep.flight_frames == 0 and rep.contact_frames > 20
    assert rep.contact_frames_by_end_effector["lhand"] == 0 and rep.contact_frames_by_end_effector["lfoot"] > 20
    assert rep.contact_wrench_residual_max_n < 1.0
    assert rep.torque_ratio_max < 0.85
    assert rep.passes(0.85, 0.02 * rep.body_weight_n, 2.0)
    assert rep.body_weight_n == pytest.approx(fb.mass_kg * 9.81)
    fields = {f.name for f in dataclasses.fields(rep)}
    assert {"torque_ratio_max", "torque_ratio_by_joint", "worst_joint", "flight_root_force_max_n",
            "flight_root_torque_max_nm", "contact_wrench_residual_max_n", "joints_over_limit"} <= fields
    assert set(rep.torque_ratio_by_joint) == set(g1_model.G1_JOINT_NAMES)
    json.dumps(dataclasses.asdict(rep), allow_nan=False)


@needs_g1
def test_free_fall_has_zero_root_wrench_and_hover_does_not(rig, fb):
    from humanoid_mimic_rl import g1_cmu_feasibility as gf

    fps, n = 50.0, 30
    base = rig.home_q.copy()
    t = np.arange(n) / fps
    fall = np.tile(base, (n, 1))
    fall[:, 2] = 3.0 - 0.5 * 9.81 * t**2  # ballistic; the feet stay above the floor for the whole 0.58 s
    hover = np.tile(base, (n, 1))
    hover[:, 2] = 3.0
    r_fall = gf.check_clip(fb, fall, fps)
    r_hover = gf.check_clip(fb, hover, fps)
    assert r_fall.flight_frames > 20 and r_fall.flight_root_force_max_n < 1.0
    assert r_hover.flight_root_force_max_n == pytest.approx(r_hover.body_weight_n, rel=1e-3)
    assert not r_hover.passes(0.85, 0.02 * r_hover.body_weight_n, 2.0)


@needs_g1
def test_ballistic_flight_fix_zeroes_com_force(rig, fb):
    from humanoid_mimic_rl import g1_cmu_feasibility as gf

    fps, n = 50.0, 60
    q = np.tile(rig.home_q, (n, 1))
    q[:, 2] = 0.785 - rig_min_clearance(rig, rig.home_q)
    q[20:40, 2] += 0.3 * np.sin(np.linspace(0.0, np.pi, 20)) ** 0.5  # a non-ballistic hop
    before = gf.check_clip(fb, q, fps)
    fixed, segments = gf.ballistic_flight_fix(fb, q, fps)
    after = gf.check_clip(fb, fixed, fps)
    assert segments == 1 and before.flight_frames > 5
    assert after.flight_root_force_max_n < 0.02 * after.body_weight_n < before.flight_root_force_max_n


@needs_g1
def test_contact_lp_unilateral_and_friction(rig, fb):
    """A sideways push beyond mu * N cannot be supplied by the feet; the LP reports it as residual."""
    from humanoid_mimic_rl import g1_cmu_feasibility as gf

    q = _home_clip(rig, 1)[0]
    geo = gf.contact_geometry(fb)
    f = fb.inverse(q, np.zeros(fb.nv), np.zeros(fb.nv))
    basis = gf.contact_basis(fb, geo, [0, 1], g1_model.FOOT_FRICTION)
    cap = fb.force_limit
    resid, ratio, _ = gf.solve_frame(f, basis, cap, cap)
    assert np.linalg.norm(resid) < 1e-6 and ratio < 0.85
    f_push = f.copy()
    f_push[0] += 3.0 * fb.mass_kg * 9.81  # needs F_x = 3 BW with only ~1 BW of normal force
    resid, _, _ = gf.solve_frame(f_push, basis, cap, cap)
    assert np.linalg.norm(resid[:3]) > 1.5 * fb.mass_kg * 9.81


@needs_g1
@needs_cmu
def test_retarget_short_clip_schema_and_contacts(rig, fb, tmp_path):
    from humanoid_mimic_rl import g1_cmu_feasibility as gf
    from humanoid_mimic_rl import g1_retarget as rt

    raw = Path(_CMU)
    sk = cmu_mocap.parse_asf(raw / "88.asf")
    full = cmu_mocap.parse_amc(raw / "88_06.amc", sk)
    short = cmu_mocap.Motion(full.root[:70], {k: v[:70] for k, v in full.angles_deg.items()})
    res = rt.retarget(rig, sk, short, 60.0)
    q = res.qpos
    assert q.shape == (res.human.num_frames, 36) and res.fps_hz == 50.0
    assert 0.5 < res.scale < 1.0
    assert np.all(np.isfinite(q))
    np.testing.assert_allclose(np.linalg.norm(q[:, 3:7], axis=1), 1.0, atol=1e-6)
    limits = rt.joint_limit_report(rig, q)
    assert limits["violating_frames"] == 0
    assert np.all(q[:, 7:] >= rig.lo - 1e-9) and np.all(q[:, 7:] <= rig.hi + 1e-9)
    quality = rt.contact_quality(rig, q, res.human_contacts)
    assert quality["max_penetration_cm"] < 2.0
    for run in quality["runs"]:
        assert run["max_height_error_cm"] < 2.5 and run["max_slip_cm"] < 3.0
    errors = rt.keypoint_errors_cm(rig, q, res.human)
    assert set(errors) == set(rt.ERROR_KEYPOINTS) and all(v["max_cm"] < 40.0 for v in errors.values())
    # the clip round-trips through the repo's clip format and keeps the licence
    flags = gf.contact_flags(fb, gf.contact_geometry(fb), q)
    clip = rt.to_clip(rig, q, res.fps_hz, flags[:, :2].any(axis=1), cmu_mocap.LICENCE, "test")
    path = tmp_path / "c.npz"
    reference.save_clip(path, clip)
    back = reference.load_clip(path)
    assert back.joint_names == g1_model.G1_JOINT_NAMES and back.site_names == g1_model.END_EFFECTOR_SITES
    assert back.foot_contact.dtype == bool and back.foot_contact.shape == (clip.num_frames,)
    assert "free for all uses" in back.licence
