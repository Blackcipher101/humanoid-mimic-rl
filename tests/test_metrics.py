import numpy as np
import pytest

from humanoid_mimic_rl import metrics, quat

Y = np.array([0.0, 1.0, 0.0])
DT = 0.02


def test_alive_mask_includes_first_done_only():
    done = np.array([[0, 0, 1, 0, 1], [0, 0, 0, 0, 0]])
    np.testing.assert_array_equal(
        metrics.alive_mask(done), [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    )
    np.testing.assert_array_equal(metrics.episode_lengths(done), [3, 5])
    np.testing.assert_array_equal(metrics.terminated_early(done), [True, False])
    np.testing.assert_allclose(metrics.episode_returns(np.ones((2, 5)), done), [3, 5])


def test_cumulative_pitch_counts_full_backflip():
    pitch = np.linspace(0.0, -2 * np.pi, 60)
    q = quat.quat_from_axis_angle(Y, pitch, np)
    cum = metrics.cumulative_pitch_rad(q[None])
    np.testing.assert_allclose(cum[0], pitch, atol=1e-9)


def test_cumulative_pitch_ignores_yaw():
    yaw = quat.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.linspace(0, 1.0, 30), np)
    np.testing.assert_allclose(metrics.cumulative_pitch_rad(yaw[None]), 0.0, atol=1e-12)


def _flip_inputs(n_steps: int = 80, final_pitch: float = -2 * np.pi):
    pitch = np.concatenate([np.linspace(0.0, final_pitch, 40), np.full(n_steps - 40, final_pitch)])
    q = quat.quat_from_axis_angle(Y, pitch, np)[None]
    up = np.cos(pitch)[None]
    h = np.full((1, n_steps), 0.62)
    contact = np.ones((1, n_steps, 2), dtype=bool)
    return q, up, h, contact


def test_flip_landing_success():
    q, up, h, contact = _flip_inputs()
    done = np.zeros((1, 80))
    out = metrics.summarize_flip(np.ones((1, 80)), done, q, up, h, contact, DT, 0.5, 0.8, 0.5)
    assert out["landing_success_rate"] == 1.0 and out["fall_rate"] == 0.0
    assert out["max_backward_rotation_rad_mean"] == pytest.approx(2 * np.pi)


def test_flip_failures():
    # Under-rotated by 90 deg: lands on its back.
    q, up, h, contact = _flip_inputs(final_pitch=-1.5 * np.pi)
    out = metrics.summarize_flip(np.ones((1, 80)), np.zeros((1, 80)), q, up, h, contact, DT, 0.5, 0.8, 0.5)
    assert out["landing_success_rate"] == 0.0 and out["fall_rate"] == 1.0
    # Full rotation but terminated early (head hit the ground).
    q, up, h, contact = _flip_inputs()
    done = np.zeros((1, 80))
    done[0, 70] = 1
    out = metrics.summarize_flip(np.ones((1, 80)), done, q, up, h, contact, DT, 0.5, 0.8, 0.5)
    assert out["landing_success_rate"] == 0.0 and out["terminated_early_rate"] == 1.0


def test_walk_metrics():
    n, t = 2, 50
    vel = np.zeros((n, t, 3))
    vel[..., 0] = 0.4
    cmd = np.zeros((n, t, 3))
    cmd[..., 0] = 0.5
    up = np.ones((n, t))
    up[1, 30:] = 0.2  # second episode falls
    h = np.full((n, t), 0.65)
    out = metrics.summarize_walk(np.ones((n, t)), np.zeros((n, t)), vel, cmd, up, h, DT, metrics.FallCriteria())
    assert out["fall_rate"] == 0.5 and out["success_rate"] == 0.5
    assert out["forward_speed_abs_error_mps"] == pytest.approx(0.1)
    assert out["commanded_forward_speed_mps"] == 0.5


def test_first_sustained():
    flag = np.array([[0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(metrics.first_sustained(flag, 3), [6, -1])


def test_getup_metrics():
    t = 200  # 4 s
    standing = np.zeros((3, t))
    standing[0, 100:] = 1  # stands at 2 s and stays
    standing[1, 100:150] = 1  # stands, then falls again
    out = metrics.summarize_getup(np.ones((3, t)), np.zeros((3, t)), standing, DT, 1.0)
    assert out["success_rate"] == pytest.approx(1 / 3)
    assert out["stood_up_rate"] == pytest.approx(2 / 3)
    assert out["fall_rate"] == pytest.approx(1 / 3)
    assert out["time_to_stand_s_median"] == pytest.approx(2.0)
