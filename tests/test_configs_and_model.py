import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_mimic_rl import configs, t1_model

_FEET_ONLY_SNIPPET = """
<mujoco model="T1">
  <default>
    <default class="t1">
      <geom contype="0" conaffinity="0"/>
      <default class="collision">
        <geom group="3"/>
        <default class="foot"><geom contype="4" conaffinity="4"/></default>
      </default>
    </default>
  </default>
  <worldbody>
    <body name="Trunk">
      <geom class="collision" type="box" size="0.1 0.1 0.1"/>
      <geom class="collision" type="cylinder" size="0.05 0.08"/>
      <geom class="collision" type="cylinder" size="0.03 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("task", sorted(configs.TASKS))
@pytest.mark.parametrize("profile", ["smoke", "full"])
def test_ppo_profiles_satisfy_brax_constraints(task, profile):
    configs.get_task(task).profile(profile).validate()


def test_full_budget_bounds():
    for spec in configs.TASKS.values():
        assert 50_000_000 <= spec.full.num_timesteps <= 200_000_000
        assert spec.smoke.num_timesteps <= 2_000_000


def test_invalid_divisibility_rejected():
    with pytest.raises(ValueError, match="divisible"):
        configs.PPOConfig(num_timesteps=10**8, num_evals=5, num_envs=1000).validate()


def test_unknown_task_and_profile():
    with pytest.raises(ValueError):
        configs.get_task("dance")
    with pytest.raises(ValueError):
        configs.get_task("walk").profile("medium")


def test_brax_kwargs_excludes_network():
    kw = configs.get_task("walk").full.brax_kwargs()
    assert "network" not in kw and kw["num_envs"] == 8192


def test_full_collision_patch():
    import xml.etree.ElementTree as ET

    root = ET.fromstring(t1_model.make_full_collision_xml(_FEET_ONLY_SNIPPET))
    coll = [d for d in root.iter("default") if d.get("class") == "collision"][0].find("geom")
    assert (coll.get("contype"), coll.get("conaffinity")) == ("2", "1")
    foot = [d for d in root.iter("default") if d.get("class") == "foot"][0].find("geom")
    assert foot.get("contype") == "4"  # feet boxes keep their own filter
    geoms = [g for g in root.iter("geom") if g.get("class") == "collision"]
    assert [g.get("type") for g in geoms] == ["box", "capsule", "capsule"]
    assert geoms[1].get("size") == "0.05 0.03"  # same end-to-end extent
    assert geoms[2].get("size") == "0.03 0"


def test_scene_include_rewrite():
    scene = '<mujoco><include file="t1_mjx_feetonly.xml"/><worldbody/></mujoco>'
    out = t1_model.make_scene_xml(scene, "x.xml")
    assert 'file="x.xml"' in out
    with pytest.raises(ValueError):
        t1_model.make_scene_xml("<mujoco/>", "x.xml")


def test_constant_tables_consistent():
    assert len(t1_model.T1_JOINT_NAMES) == len(t1_model.T1_HOME_JOINT_POS) == len(t1_model.T1_JOINT_RANGES_RAD)
    lo, hi = t1_model.joint_ranges().T
    home = t1_model.home_pose()
    assert np.all(lo <= home) and np.all(home <= hi)


# Optional: compile the real model. Set T1_XML_DIR to playground's
# locomotion/t1/xmls and T1_MESH_DIR to menagerie's booster_t1/assets.
_XML_DIR = os.environ.get("T1_XML_DIR")
_MESH_DIR = os.environ.get("T1_MESH_DIR")


@pytest.mark.skipif(not (_XML_DIR and _MESH_DIR), reason="T1_XML_DIR / T1_MESH_DIR not set")
def test_real_t1_full_collision_model():
    mujoco = pytest.importorskip("mujoco")
    from humanoid_mimic_rl.mj_kinematics import MjKinematics

    assets = {p.name: p.read_bytes() for p in [*Path(_XML_DIR).glob("*.xml"), *Path(_MESH_DIR).glob("*.stl")]}
    scene, assets = t1_model.full_collision_flat_scene(assets)
    m = mujoco.MjModel.from_xml_string(scene, assets=assets)
    assert tuple(m.joint(i).name for i in range(1, m.njnt)) == t1_model.T1_JOINT_NAMES
    np.testing.assert_allclose(m.keyframe("home").qpos[7:], t1_model.home_pose())
    np.testing.assert_allclose(m.jnt_range[1:], t1_model.joint_ranges())
    floor = m.geom("floor").id
    hits_floor = {
        m.body(m.geom_bodyid[g]).name
        for g in range(m.ngeom)
        if g != floor and ((m.geom_contype[g] & m.geom_conaffinity[floor]) or (m.geom_conaffinity[g] & m.geom_contype[floor]))
    }
    assert {"Trunk", "H2", "Shank_Left", "AL3", "left_foot_link"} <= hits_floor
    kin = MjKinematics(m, "Trunk", t1_model.T1_FOOT_SPHERE_GEOMS, t1_model.T1_END_EFFECTOR_SITES)
    fk = kin.forward(m.keyframe("home").qpos.copy())
    # Home keyframe stands on the floor: lowest foot point within 1 cm of z = 0.
    assert abs(fk.foot_points[:, 2].min()) < 0.01
