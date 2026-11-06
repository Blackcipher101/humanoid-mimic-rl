import numpy as np
import pytest

from humanoid_mimic_rl import quat
from humanoid_mimic_rl.supermimic import state as st

N_J = 4


def _random_state(rng: np.random.Generator, batch: int) -> tuple[np.ndarray, np.ndarray]:
    q = rng.normal(size=(batch, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    qpos = np.concatenate([rng.normal(size=(batch, 3)), q, rng.normal(size=(batch, N_J))], axis=-1)
    qvel = rng.normal(size=(batch, 6 + N_J))
    return qpos, qvel


def test_dimensions():
    rng = np.random.default_rng(0)
    qpos, qvel = _random_state(rng, 5)
    assert st.features(qpos, qvel, np.zeros((5, N_J)), np).shape == (5, st.num_features(N_J))
    qpos2, qvel2 = _random_state(rng, 5)
    assert st.targets_from_transition(qpos, qvel, qpos2, qvel2, np).shape == (5, st.num_outputs(N_J))
    assert st.state_error(qpos, qvel, qpos2, qvel2, np).shape == (5, st.num_outputs(N_J))


def test_integrate_inverts_targets():
    rng = np.random.default_rng(1)
    qpos, qvel = _random_state(rng, 64)
    qpos2, qvel2 = _random_state(rng, 64)
    # Keep the per-step rotation below pi so the log map is unambiguous.
    small = quat.rotvec_to_quat(rng.uniform(-0.5, 0.5, size=(64, 3)), np)
    qpos2[:, 3:7] = quat.quat_mul(qpos[:, 3:7], small, np)
    out = st.targets_from_transition(qpos, qvel, qpos2, qvel2, np)
    qpos_hat, qvel_hat = st.integrate(qpos, qvel, out, np)
    np.testing.assert_allclose(qpos_hat[:, :3], qpos2[:, :3], atol=1e-9)
    np.testing.assert_allclose(qpos_hat[:, 7:], qpos2[:, 7:], atol=1e-9)
    np.testing.assert_allclose(quat.quat_angle_between(qpos_hat[:, 3:7], qpos2[:, 3:7], np), 0.0, atol=1e-6)
    np.testing.assert_allclose(qvel_hat, qvel2, atol=1e-9)
    np.testing.assert_allclose(st.state_error(qpos_hat, qvel_hat, qpos2, qvel2, np), 0.0, atol=1e-6)


def test_features_invariant_to_yaw_and_horizontal_translation():
    rng = np.random.default_rng(2)
    qpos, qvel = _random_state(rng, 8)
    yaw = quat.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), 1.1, np)
    qpos2 = qpos.copy()
    qpos2[:, 0:2] += np.array([3.0, -2.0])
    qpos2[:, 3:7] = quat.quat_mul(np.tile(yaw, (8, 1)), qpos[:, 3:7], np)
    qvel2 = qvel.copy()
    qvel2[:, 0:3] = quat.quat_rotate(np.tile(yaw, (8, 1)), qvel[:, 0:3], np)  # world velocity rotates too
    u = rng.normal(size=(8, N_J))
    np.testing.assert_allclose(st.features(qpos, qvel, u, np), st.features(qpos2, qvel2, u, np), atol=1e-9)


def test_gravity_body_upright_and_inverted():
    up = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(st.gravity_body(up, np), [0, 0, -1])
    flipped = quat.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi, np)
    np.testing.assert_allclose(st.gravity_body(flipped, np), [0, 0, 1], atol=1e-12)


def test_actor_obs_layout():
    qpos = np.concatenate([[0, 0, 0.6], [1, 0, 0, 0], np.full(N_J, 0.5)])
    qvel = np.concatenate([[1, 0, 0], [0.1, 0.2, 0.3], np.full(N_J, 2.0)])
    obs = st.actor_obs(qpos, qvel, np.full(N_J, -1.0), 1.0, np.full(N_J, 0.25), 2.0, np)
    assert obs.shape == (3 + 3 + 3 * N_J + 3,)
    np.testing.assert_allclose(obs[0:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(obs[3:6], [0, 0, -1])
    np.testing.assert_allclose(obs[6 : 6 + N_J], 0.25)
    np.testing.assert_allclose(obs[-3:], [0.5, 0.0, -1.0], atol=1e-12)


def test_window_valid_mask():
    done = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(
        st.window_valid_mask(done, np),
        [[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]],
    )


@pytest.mark.parametrize(
    "ptr, size, expected",
    [(5, 5, [0, 1, 2]), (2, 8, [2, 3, 4]), (0, 8, [0, 1, 2])],
)
def test_ring_rows(ptr, size, expected):
    np.testing.assert_array_equal(st.ring_rows(ptr, size, 8, np.array([0, 1, 2]), np), expected)
