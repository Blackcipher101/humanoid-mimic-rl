"""G1 model patch, floating-base dynamics and flip-optimiser physics.

The dynamics tests use a small synthetic biped (MJCF below) and need the
``mujoco`` and ``scipy`` packages; the G1 tests need the real model files via
``G1_XML_DIR`` (playground ``locomotion/g1/xmls``) and ``G1_MESH_DIR``
(menagerie ``unitree_g1/assets``) and otherwise skip.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from humanoid_mimic_rl import g1_model, reference

_G1_SNIPPET = """
<mujoco>
  <worldbody>
    <body name="pelvis"><freejoint/>
      <geom name="left_thigh" class="collision" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.2"/>
      <geom name="right_thigh" class="collision" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.2"/>
      <geom name="left_shin" class="collision" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.2"/>
      <geom name="right_shin" class="collision" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.2"/>
      <geom name="left_hand_collision" class="collision" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.2"/>
      <geom name="right_hand_collision" class="collision" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.2"/>
      <body name="torso_link"/>
    </body>
  </worldbody>
  <contact><pair geom1="left_thigh" geom2="right_thigh"/></contact>
</mujoco>
"""


def test_full_collision_patch_adds_colliders_and_floor_pairs():
    root = ET.fromstring(g1_model.make_full_collision_xml(_G1_SNIPPET))
    names = {g.get("name"): g for g in root.iter("geom")}
    for added in g1_model.FALL_GEOMS:
        assert names[added].get("class") == "collision"
    pairs = [p for p in root.iter("pair") if p.get("geom2") == "floor"]
    assert {p.get("geom1") for p in pairs} == set(g1_model.FLOOR_PAIR_GEOMS)
    assert all(p.get("condim") == "3" for p in pairs)
    with pytest.raises(ValueError, match="already has"):
        g1_model.make_full_collision_xml(ET.tostring(root, encoding="unicode"))


def test_scene_include_rewrite():
    out = g1_model.make_scene_xml('<mujoco><include file="g1_mjx_feetonly.xml"/></mujoco>', "x.xml")
    assert 'file="x.xml"' in out


def test_constant_tables():
    assert len(g1_model.G1_JOINT_NAMES) == len(g1_model.G1_HOME_JOINT_POS) == 29


# ----------------------------------------------------------------------
# Synthetic biped for the dynamics tests: 3 bodies, sagittal hinges.
# ----------------------------------------------------------------------

_BIPED = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.01"/>
    <body name="base" pos="0 0 0.6">
      <freejoint/>
      <inertial pos="0 0 0.1" mass="10" diaginertia="0.3 0.3 0.1"/>
      <geom name="trunk" type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
      <body name="l_leg" pos="0 0.1 0">
        <joint name="l_hip" axis="0 1 0" range="-2 2" actuatorfrcrange="-200 200"/>
        <inertial pos="0 0 -0.25" mass="2" diaginertia="0.04 0.04 0.01"/>
        <body name="l_foot" pos="0 0 -0.5">
          <joint name="l_ankle" axis="0 1 0" range="-1 1" actuatorfrcrange="-100 100"/>
          <inertial pos="0.03 0 -0.02" mass="0.5" diaginertia="0.002 0.002 0.001"/>
          <geom name="l_sole" type="box" size="0.08 0.03 0.01" pos="0.03 0 -0.03"/>
        </body>
      </body>
      <body name="r_leg" pos="0 -0.1 0">
        <joint name="r_hip" axis="0 1 0" range="-2 2" actuatorfrcrange="-200 200"/>
        <inertial pos="0 0 -0.25" mass="2" diaginertia="0.04 0.04 0.01"/>
        <body name="r_foot" pos="0 0 -0.5">
          <joint name="r_ankle" axis="0 1 0" range="-1 1" actuatorfrcrange="-100 100"/>
          <inertial pos="0.03 0 -0.02" mass="0.5" diaginertia="0.002 0.002 0.001"/>
          <geom name="r_sole" type="box" size="0.08 0.03 0.01" pos="0.03 0 -0.03"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="l_hip" kp="500" ctrlrange="-2 2"/>
    <position joint="l_ankle" kp="500" ctrlrange="-1 1"/>
    <position joint="r_hip" kp="500" ctrlrange="-2 2"/>
    <position joint="r_ankle" kp="500" ctrlrange="-1 1"/>
  </actuator>
  <keyframe><key name="home" qpos="0 0 0.64 1 0 0 0 0 0 0 0"/></keyframe>
</mujoco>
"""


@pytest.fixture()
def biped():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    from humanoid_mimic_rl import flip_dynamics as fd

    m = mujoco.MjModel.from_xml_string(_BIPED)
    feet = fd.FootGeometry(("l_foot", "r_foot"), (0.03, 0.0, -0.04), 0.08, 0.03)
    return fd.FloatingBaseModel(m, feet, ["trunk"])


def _problem(fb, rate_hz=100.0):
    from humanoid_mimic_rl import flip_opt

    setup = flip_opt.FlipSetup(
        groups={"hip": (0, 2), "ankle": (1, 3)},
        home_qpos=fb.m.keyframe("home").qpos.copy(),
        knot_times_s=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]),
        duration_s=1.4,
        sample_rate_hz=rate_hz,
        takeoff_bounds_s=(0.3, 0.7),
        min_flight_s=0.1,
        max_flight_s=0.6,
    )
    return flip_opt.FlipProblem(fb, setup)


def test_static_stance_wrench_supports_weight(biped):
    fb = biped
    q = fb.m.keyframe("home").qpos.copy()
    f = fb.inverse(q, np.zeros(fb.nv), np.zeros(fb.nv))
    w, tau = fb.foot_wrench(f)
    weight = fb.mass_kg * 9.81
    assert w[2] == pytest.approx(weight / 2, rel=1e-6)
    np.testing.assert_allclose(w[[0, 1, 5]], 0.0, atol=1e-6)
    # Centre of pressure under the CoM: CoM x = (10*0 + 4*0 + 1*0.03)/15 = 0.002, sole centre x = 0.03.
    assert -w[4] / w[2] == pytest.approx(0.002 - 0.03, abs=1e-6)
    assert np.all(np.isfinite(tau))


def test_root_accel_free_fall(biped):
    fb = biped
    q = fb.m.keyframe("home").qpos.copy()
    a = fb.root_accel(q, np.zeros(fb.nv), np.zeros(fb.m.nu))
    np.testing.assert_allclose(a, [0, 0, -9.81, 0, 0, 0], atol=1e-9)


def test_actuator_caps_pd_authority(biped):
    cap_pos, cap_neg = biped.actuator_caps(np.array([0.9, 0.9, 0.0, 0.0]))
    assert cap_pos[1] == pytest.approx(50.0)  # kp 500 * (1 - 0.9)
    assert cap_neg[1] == pytest.approx(100.0)  # force limit binds
    assert cap_pos[0] == pytest.approx(200.0)


def test_torque_motors_cap_is_peak_torque(biped):
    from humanoid_mimic_rl import g1_model

    peak = g1_model.flip_peak_torques(biped.m)
    np.testing.assert_allclose(peak, [120.0, 100.0, 120.0, 100.0])  # min(force limit, 120 N m)
    g1_model.to_torque_motors(biped.m, peak)
    np.testing.assert_allclose(biped.m.actuator_ctrlrange[:, 1], peak)
    np.testing.assert_allclose(biped.m.jnt_actfrcrange[1:, 1], peak)
    fb = type(biped)(biped.m, biped.feet, ["trunk"])
    for q in (np.zeros(4), np.array([1.95, 0.99, -1.95, -0.99])):  # also right at the joint limits
        cap_pos, cap_neg = fb.actuator_caps(q)
        np.testing.assert_allclose(cap_pos, peak)
        np.testing.assert_allclose(cap_neg, peak)
    with pytest.raises(ValueError):
        g1_model.to_torque_motors(biped.m, np.zeros(4))


def test_torque_motor_applies_ctrl_as_torque(biped):
    mujoco = pytest.importorskip("mujoco")
    from humanoid_mimic_rl import g1_model

    m = mujoco.MjModel.from_xml_string(_BIPED)
    g1_model.to_torque_motors(m, g1_model.flip_peak_torques(m))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = [50.0, -500.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(d.actuator_force, [50.0, -100.0, 0.0, 0.0])  # clamped to the peak


def test_ground_cop_and_support_region():
    from humanoid_mimic_rl import flip_dynamics as fd

    centre = np.array([0.1, 0.0, 0.0])
    w = np.array([0.0, 0.0, 100.0, 0.0, -2.0, 0.0])  # tau_y = -2 N m -> CoP 2 cm ahead
    np.testing.assert_allclose(fd.ground_cop(centre, w), [0.12, 0.0])
    corners = np.array([[0.0, -0.03, 0.0], [0.0, 0.03, 0.0], [0.2, -0.03, 0.0], [0.2, 0.03, 0.0]])
    assert fd.cop_outside_support(np.array([0.12, 0.0]), corners) == 0.0
    assert fd.cop_outside_support(np.array([0.25, 0.0]), corners) == pytest.approx(0.05)
    heel_up = corners.copy()
    heel_up[:2, 2] = 0.05  # heel corners lifted: support is the toe line x = 0.2
    assert fd.cop_outside_support(np.array([0.12, 0.0]), heel_up) == pytest.approx(0.08)


def test_toe_off_pivots_about_the_toe_edge(biped):
    p = _problem(biped)
    x = p.pack(np.array([[0.3, 0.2], [-0.4, 0.3], [0.2, -0.2], [0.0, 0.1]]), 0.5, heel_max_rad=0.4, heel_lead_s=0.2)
    assert p.heel_angle(x, 0.25) == 0.0 and p.heel_angle(x, 0.5) == pytest.approx(0.4)
    spl = p.joint_spline(x)
    q, _, _ = p.stance_state(spl, 0.5, p.toe0, lambda t: p.heel_angle(x, t))
    pos, rot = biped.foot_pose(q, 0)
    np.testing.assert_allclose(pos + rot @ p.toe_local, p.toe0, atol=1e-9)  # toe edge stays put
    assert np.arctan2(rot[0, 2], rot[0, 0]) == pytest.approx(0.4)  # heel up = positive foot pitch


def test_flight_is_ballistic_and_conserves_angular_momentum(biped):
    p = _problem(biped)
    x = p.pack(np.array([[0.3, 0.2], [-0.8, 0.5], [0.6, -0.4], [-0.2, 0.1]]), 0.5)
    spl = p.joint_spline(x)
    q0, v0, _ = p.stance_state(spl, 0.5, p.toe0)
    y0 = np.concatenate([q0[0:3], q0[3:7], v0[0:3] + [0.0, 0.0, 3.0], v0[3:6] + [0.0, -4.0, 0.0]])
    ts, ys, _ = p.flight(spl, 0.5, y0)
    com_v, ang = [], []
    for t, y in zip(ts, ys):
        qj, qdj, _ = p.joints_at(spl, t)
        cv, am = p._momentum(np.concatenate([y[0:3], y[3:7], qj]), np.concatenate([y[7:10], y[10:13], qdj]))
        com_v.append(cv)
        ang.append(am)
    com_v, ang, ts = np.array(com_v), np.array(ang), np.asarray(ts)
    steps = np.diff(ts) > 1e-6
    acc = np.diff(com_v, axis=0)[steps] / np.diff(ts)[steps][:, None]
    np.testing.assert_allclose(acc[:, 2], -9.81, atol=1e-4)
    np.testing.assert_allclose(acc[:, 0], 0.0, atol=1e-4)
    np.testing.assert_allclose(ang[:, 1], ang[0, 1], atol=1e-5 * max(1.0, abs(ang[0, 1])))


def test_stance_keeps_foot_flat_and_fixed(biped):
    p = _problem(biped)
    x = p.pack(np.array([[0.3, 0.2], [-0.4, 0.3], [0.2, -0.2], [0.0, 0.1]]), 0.5)
    spl = p.joint_spline(x)
    for t in (0.0, 0.25, 0.45):
        q, _, _ = p.stance_state(spl, t, p.toe0)  # flat foot: toe anchor <=> foot origin at anchor0
        pos, rot = biped.foot_pose(q, 0)
        np.testing.assert_allclose(pos, p.anchor0, atol=1e-9)
        np.testing.assert_allclose(rot, np.eye(3), atol=1e-9)


def test_check_clip_flight_wrench_is_zero_for_simulated_flight(biped):
    from humanoid_mimic_rl import flip_dynamics as fd

    p = _problem(biped, rate_hz=200.0)
    x = p.pack(np.array([[0.3, 0.2], [-0.8, 0.5], [0.6, -0.4], [-0.2, 0.1]]), 0.5)
    r = p.rollout(x)
    clip = reference.ReferenceClip(
        fps_hz=200.0, qpos=r.qpos, qvel=r.qvel, joint_names=("l_hip", "l_ankle", "r_hip", "r_ankle"),
        site_names=(), site_pos=np.zeros((len(r.qpos), 0, 3)), licence="test", source="test",
        foot_contact=r.contact,
    )
    rep = fd.check_clip(biped, clip, r.contact, clip.joint_names)
    assert rep.flight_frames > 10
    assert rep.flight_root_force_max_n < 0.02 * rep.body_weight_n
    assert rep.flight_root_torque_max_nm < 0.5


# ----------------------------------------------------------------------
# Real G1 model (optional).
# ----------------------------------------------------------------------

_XML = os.environ.get("G1_XML_DIR")
_MESH = os.environ.get("G1_MESH_DIR")


def _g1_assets() -> dict[str, bytes]:
    xml = Path(_XML)
    files = [*xml.glob("*.xml"), *(xml / "assets").glob("*"), *Path(_MESH).glob("*")]
    return {f.name: f.read_bytes() for f in files if f.is_file()}


@pytest.mark.skipif(not (_XML and _MESH), reason="G1_XML_DIR / G1_MESH_DIR not set")
def test_real_g1_full_collision_model():
    mujoco = pytest.importorskip("mujoco")
    scene, assets = g1_model.full_collision_flat_scene(_g1_assets())
    m = mujoco.MjModel.from_xml_string(scene, assets=assets)
    assert tuple(m.joint(i).name for i in range(1, m.njnt)) == g1_model.G1_JOINT_NAMES
    np.testing.assert_allclose(m.keyframe("home").qpos[7:], g1_model.home_pose())
    assert m.npair == 5 + len(g1_model.FLOOR_PAIR_GEOMS)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[2] = 0.3
    d.qpos[3:7] = [np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0]  # lying on the back
    for _ in range(1500):
        mujoco.mj_step(m, d)
    touching = {m.geom(d.contact[i].geom1).name for i in range(d.ncon)} | {
        m.geom(d.contact[i].geom2).name for i in range(d.ncon)
    }
    assert "pelvis_collision" in touching or "torso_collision" in touching


@pytest.mark.skipif(not (_XML and _MESH), reason="G1_XML_DIR / G1_MESH_DIR not set")
def test_shipped_g1_backflip_solution_is_feasible():
    pytest.importorskip("scipy")
    from humanoid_mimic_rl import flip_dynamics as fd
    from humanoid_mimic_rl import flip_opt, g1_backflip

    if not g1_backflip.SOLUTION_PATH.is_file():
        pytest.skip("no optimised G1 backflip solution yet (run scripts/make_g1_backflip.py --optimize)")
    x, scale = g1_backflip.load_solution(g1_backflip.SOLUTION_PATH)
    problem = g1_backflip.make_problem(_g1_assets(), 400.0, scale)
    r = problem.rollout(x)
    assert r.landed
    clip = flip_opt.to_clip(problem, r, g1_model.G1_JOINT_NAMES, g1_model.END_EFFECTOR_SITES, "t", "t")
    rep = fd.check_clip(problem.fb, clip, clip.foot_contact, g1_model.G1_JOINT_NAMES)
    assert rep.passes(0.85, g1_model.FOOT_FRICTION, 0.02 * rep.body_weight_n, 2.0), rep
    pitch = np.unwrap(2 * np.arctan2(clip.qpos[:, 5], clip.qpos[:, 3]))
    assert pitch[-1] - pitch[0] == pytest.approx(-2 * np.pi, abs=0.05)


def test_torque_scale_multiplies_peaks_and_is_labelled(biped):
    from humanoid_mimic_rl import g1_model

    np.testing.assert_allclose(g1_model.flip_peak_torques(biped.m, 3.0), [360.0, 300.0, 360.0, 300.0])
    with pytest.raises(ValueError):
        g1_model.flip_peak_torques(biped.m, 0.0)
    assert "modified model" in g1_model.torque_scale_label(3.0)
    assert "3x published peak torque" in g1_model.torque_scale_label(3.0)
    assert "modified" not in g1_model.torque_scale_label(1.0)


def test_resolve_torque_scale_from_clip_report(tmp_path):
    import json

    from humanoid_mimic_rl import g1_model

    clip = tmp_path / "flip.npz"
    with pytest.raises(ValueError, match="not found"):
        g1_model.resolve_torque_scale(clip, None)
    report = g1_model.clip_report_path(clip)
    report.write_text(json.dumps({"torque_scale": 4.0, "passed": True}))
    assert g1_model.resolve_torque_scale(clip, None) == 4.0
    assert g1_model.resolve_torque_scale(clip, 4.0) == 4.0
    with pytest.raises(ValueError, match="!="):
        g1_model.resolve_torque_scale(clip, 3.0)
    report.write_text(json.dumps({"torque_scale": 4.0, "passed": False}))
    with pytest.raises(ValueError, match="FAILED"):
        g1_model.resolve_torque_scale(clip, None)


def test_check_clip_guards_vanishing_normal_force(biped):
    """Frames flagged as stance with ~zero normal force are counted, not divided by."""
    import dataclasses
    import json

    from humanoid_mimic_rl import flip_dynamics as fd

    p = _problem(biped, rate_hz=200.0)
    x = p.pack(np.array([[0.3, 0.2], [-0.8, 0.5], [0.6, -0.4], [-0.2, 0.1]]), 0.5)
    r = p.rollout(x)
    contact = np.ones_like(r.contact)  # wrongly claim contact during the flight
    clip = reference.ReferenceClip(
        fps_hz=200.0, qpos=r.qpos, qvel=r.qvel, joint_names=("l_hip", "l_ankle", "r_hip", "r_ankle"),
        site_names=(), site_pos=np.zeros((len(r.qpos), 0, 3)), licence="test", source="test",
        foot_contact=contact,
    )
    rep = fd.check_clip(biped, clip, contact, clip.joint_names)
    assert rep.stance_low_normal_frames > 10
    assert np.isfinite(rep.stance_max_friction_ratio) and np.isfinite(rep.stance_max_cop_outside_m)
    assert not rep.passes(0.85, 0.6, 1e9, 1e9)
    json.dumps(dataclasses.asdict(rep), allow_nan=False)  # report is valid JSON (no inf / NaN)
