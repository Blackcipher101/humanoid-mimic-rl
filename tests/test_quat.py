import numpy as np
import pytest

from humanoid_mimic_rl import quat

Y = np.array([0.0, 1.0, 0.0])


def _rand_quat(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_axis_angle_rotation_matches_rotation_matrix():
    q = quat.quat_from_axis_angle(Y, np.pi / 2, np)
    # +90 deg about y maps body x to world -z and body z to world +x.
    np.testing.assert_allclose(quat.quat_rotate(q, np.array([1.0, 0, 0]), np), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(quat.quat_rotate(q, np.array([0, 0, 1.0]), np), [1, 0, 0], atol=1e-12)


def test_mul_conj_identity():
    q = _rand_quat(np.random.default_rng(0), 50)
    ident = quat.quat_mul(q, quat.quat_conj(q, np), np)
    np.testing.assert_allclose(ident, np.tile([1.0, 0, 0, 0], (50, 1)), atol=1e-12)


def test_rotvec_roundtrip_and_double_cover():
    rng = np.random.default_rng(1)
    rv = rng.uniform(-1.0, 1.0, size=(100, 3)) * 2.5  # angles up to ~4.3 rad -> wrap
    q = quat.rotvec_to_quat(rv, np)
    back = quat.quat_to_rotvec(q, np)
    # Same rotation, shortest representation (angle <= pi).
    assert np.all(np.linalg.norm(back, axis=-1) <= np.pi + 1e-9)
    np.testing.assert_allclose(
        np.abs(np.sum(quat.rotvec_to_quat(back, np) * q, axis=-1)), 1.0, atol=1e-9
    )
    np.testing.assert_allclose(quat.quat_to_rotvec(-q, np), back, atol=1e-9)


def test_small_angle_is_finite():
    rv = np.array([1e-9, 0.0, 0.0])
    q = quat.rotvec_to_quat(rv, np)
    np.testing.assert_allclose(quat.quat_to_rotvec(q, np), rv, atol=1e-15)
    np.testing.assert_allclose(quat.quat_to_rotvec(np.array([1.0, 0, 0, 0]), np), 0.0)


def test_angle_between():
    q0 = quat.quat_from_axis_angle(Y, 0.0, np)
    q1 = quat.quat_from_axis_angle(Y, 0.3, np)
    assert quat.quat_angle_between(q0, q1, np) == pytest.approx(0.3)
    # Full turn is the identity rotation.
    q2 = quat.quat_from_axis_angle(Y, 2 * np.pi, np)
    assert quat.quat_angle_between(q0, q2, np) == pytest.approx(0.0, abs=1e-7)


def test_nlerp_endpoints_and_shortest_arc():
    q0 = quat.quat_from_axis_angle(Y, 0.0, np)
    q1 = -quat.quat_from_axis_angle(Y, 0.2, np)  # opposite hemisphere, same rotation
    mid = quat.quat_nlerp(q0, q1, 0.5, np)
    assert quat.quat_angle_between(q0, mid, np) == pytest.approx(0.1, abs=1e-6)
    np.testing.assert_allclose(quat.quat_nlerp(q0, q1, 0.0, np), q0, atol=1e-12)


def test_normalize_zero_returns_identity():
    np.testing.assert_allclose(quat.quat_normalize(np.zeros(4), np), [1, 0, 0, 0])
