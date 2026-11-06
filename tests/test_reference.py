from pathlib import Path

import numpy as np
import pytest

from humanoid_mimic_rl import quat, reference

Y = np.array([0.0, 1.0, 0.0])
FPS = 100.0


def _spinning_clip(n: int = 101, rate_radps: float = -6.0) -> np.ndarray:
    """Root translating at (1, 0, 2) m/s and rotating at a constant rate about y."""
    t = np.arange(n) / FPS
    pos = np.stack([t, np.zeros(n), 0.5 + 2.0 * t], axis=-1)
    q = quat.quat_from_axis_angle(Y, rate_radps * t, np)
    joints = np.stack([np.sin(t), 0.5 * t], axis=-1)
    return np.concatenate([pos, q, joints], axis=1)


def _clip(qpos: np.ndarray, **kw) -> reference.ReferenceClip:
    args = dict(
        fps_hz=FPS,
        qpos=qpos,
        qvel=reference.finite_difference_qvel(qpos, FPS),
        joint_names=("a", "b"),
        site_names=("s",),
        site_pos=np.zeros((qpos.shape[0], 1, 3)),
        licence="test licence",
        source="unit test",
    )
    args.update(kw)
    return reference.ReferenceClip(**args)


def test_finite_difference_velocity_conventions():
    qpos = _spinning_clip()
    qvel = reference.finite_difference_qvel(qpos, FPS)
    np.testing.assert_allclose(qvel[:, 0:3], np.tile([1.0, 0.0, 2.0], (len(qpos), 1)), atol=1e-9)
    # Body-frame angular velocity of a constant-rate rotation about y (also the
    # world y axis here), exact even across the +/-pi quaternion wrap.
    np.testing.assert_allclose(qvel[:, 3:6], np.tile([0.0, -6.0, 0.0], (len(qpos), 1)), atol=1e-9)
    t = np.arange(len(qpos)) / FPS
    np.testing.assert_allclose(qvel[1:-1, 6], np.cos(t[1:-1]), atol=1e-4)


def test_angular_velocity_is_body_frame():
    # Pitch by 90 deg, then spin about the *body* z axis: body-frame omega = (0, 0, w).
    n = 21
    t = np.arange(n) / FPS
    base = quat.quat_from_axis_angle(Y, np.pi / 2, np)
    spin = quat.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), 3.0 * t, np)
    q = quat.quat_mul(np.tile(base, (n, 1)), spin, np)
    qpos = np.concatenate([np.zeros((n, 3)), q, np.zeros((n, 2))], axis=1)
    qvel = reference.finite_difference_qvel(qpos, FPS)
    np.testing.assert_allclose(qvel[:, 3:6], np.tile([0.0, 0.0, 3.0], (n, 1)), atol=1e-9)


def test_save_load_roundtrip(tmp_path: Path):
    clip = _clip(_spinning_clip())
    path = tmp_path / "c.npz"
    reference.save_clip(path, clip)
    loaded = reference.load_clip(path)
    np.testing.assert_array_equal(loaded.qpos, clip.qpos)
    assert loaded.joint_names == ("a", "b") and loaded.licence == "test licence"
    assert loaded.duration_s == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda q: {"licence": " "}, "licence"),
        (lambda q: {"qpos": np.where(np.arange(q.shape[1]) == 7, np.nan, q)}, "NaN"),
        (lambda q: {"qpos": q * np.r_[np.ones(3), 2 * np.ones(4), np.ones(2)]}, "norm"),
        (lambda q: {"joint_names": ("a",)}, "joints"),
        (lambda q: {"fps_hz": 0.0}, "fps"),
    ],
)
def test_validation_rejects_bad_clips(mutate, msg):
    q = _spinning_clip()
    changes = mutate(q)
    clip = _clip(changes.pop("qpos", q), **changes)
    with pytest.raises(reference.ClipError, match=msg):
        clip.validate()


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(reference.ClipError, match="not found"):
        reference.load_clip(tmp_path / "missing.npz")


def test_sample_clip_interpolates_and_holds_final_frame():
    clip = _clip(_spinning_clip())
    q, v, s = reference.sample_clip(clip.qpos, clip.qvel, clip.site_pos, FPS, 0.255, np)
    np.testing.assert_allclose(q[0:3], [0.255, 0.0, 0.5 + 2 * 0.255], atol=1e-9)
    assert quat.quat_angle_between(q[3:7], quat.quat_from_axis_angle(Y, -6.0 * 0.255, np), np) < 1e-3
    np.testing.assert_allclose(v[0:3], [1.0, 0.0, 2.0], atol=1e-9)
    assert s.shape == (1, 3)
    q_end, v_end, _ = reference.sample_clip(clip.qpos, clip.qvel, clip.site_pos, FPS, 5.0, np)
    np.testing.assert_allclose(q_end, clip.qpos[-1], atol=1e-12)
    np.testing.assert_array_equal(v_end, 0.0)
    q0, _, _ = reference.sample_clip(clip.qpos, clip.qvel, clip.site_pos, FPS, -1.0, np)
    np.testing.assert_allclose(q0, clip.qpos[0])
