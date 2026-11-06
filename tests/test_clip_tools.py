import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_mimic_rl import quat

clip_tools = pytest.importorskip("humanoid_mimic_rl.clip_tools")  # needs mujoco


def test_yaw_align_moves_start_to_origin_heading_x():
    z = np.array([0.0, 0.0, 1.0])
    q = quat.quat_from_axis_angle(z, 1.2, np)
    n = 5
    qpos = np.zeros((n, 9))
    qpos[:, 0:3] = np.array([3.0, -2.0, 0.66]) + np.outer(np.arange(n), [np.cos(1.2), np.sin(1.2), 0.0]) * 0.1
    qpos[:, 3:7] = q
    qpos[:, 7:] = [0.1, -0.2]
    qvel = np.zeros((n, 8))
    qvel[:, 0:3] = [0.5 * np.cos(1.2), 0.5 * np.sin(1.2), 0.0]
    qvel[:, 3:6] = [0.0, 0.0, 0.3]
    p2, v2 = clip_tools.yaw_align(qpos, qvel)
    np.testing.assert_allclose(p2[0, 0:3], [0.0, 0.0, 0.66], atol=1e-12)
    np.testing.assert_allclose(p2[:, 1], 0.0, atol=1e-12)  # moves along +x only
    np.testing.assert_allclose(p2[:, 3:7], np.tile([1.0, 0, 0, 0], (n, 1)), atol=1e-12)
    np.testing.assert_allclose(v2[:, 0:3], np.tile([0.5, 0.0, 0.0], (n, 1)), atol=1e-12)
    np.testing.assert_allclose(v2[:, 3:], qvel[:, 3:])  # body rates and joint rates unchanged


_XML = os.environ.get("T1_XML_DIR")
_MESH = os.environ.get("T1_MESH_DIR")


@pytest.mark.skipif(not (_XML and _MESH), reason="T1_XML_DIR / T1_MESH_DIR not set")
def test_sanity_inverse_dynamics_static_stance_is_small():
    import mujoco

    from humanoid_mimic_rl import t1_model

    files = [*Path(_XML).glob("*.xml"), *(Path(_XML) / "assets").glob("*"), *Path(_MESH).glob("*.stl")]
    scene, assets = t1_model.full_collision_flat_scene({f.name: f.read_bytes() for f in files if f.is_file()})
    m = mujoco.MjModel.from_xml_string(scene, assets=assets)
    flags = m.opt.disableflags
    qpos = np.tile(m.keyframe("home").qpos, (5, 1))
    out = clip_tools.sanity_inverse_dynamics(m, qpos, np.zeros((5, m.nv)), 50.0)
    assert m.opt.disableflags == flags  # caller's model untouched
    assert out["max_root_force_when_no_foot_touches_n"] == 0.0
    assert out["approx_max_torque_over_force_limit"] < 0.5  # standing still needs little torque
