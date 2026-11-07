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


def test_tracking_metrics_and_flip_detection():
    from humanoid_mimic_rl import reference

    n_t = 50
    t = np.arange(n_t) / 50.0
    ref = np.zeros((n_t, 9))
    ref[:, 0] = 0.5 * t
    ref[:, 2] = 0.66
    ref[:, 3] = 1.0
    ref[:, 7:] = 0.2
    clip = reference.ReferenceClip(
        fps_hz=50.0, qpos=ref, qvel=reference.finite_difference_qvel(ref, 50.0), joint_names=("a", "b"),
        site_names=(), site_pos=np.zeros((n_t, 0, 3)), licence="test", source="test",
    )
    times = np.tile(t, (2, 1))
    sampled = reference.sample_qpos_batch(clip, times)
    np.testing.assert_allclose(sampled[0], ref, atol=1e-12)
    qpos = sampled.copy()
    qpos[:, :, 0] += 0.1  # 10 cm behind/ahead everywhere
    qpos[:, :, 7:] += 0.05
    done = np.zeros((2, n_t))
    done[1, 24] = 1.0
    out = metrics.summarize_tracking(np.ones((2, n_t)), done, qpos, sampled, 0.02)
    assert out["root_pos_error_m"] == pytest.approx(0.1)
    assert out["joint_rmse_rad"] == pytest.approx(0.05)
    assert out["root_orientation_error_rad"] == pytest.approx(0.0, abs=1e-9)
    assert out["completion_rate"] == 0.5
    assert out["survival_time_s_mean"] == pytest.approx((50 + 25) / 2 * 0.02)
    assert not metrics.is_flip_clip(ref)
    flip = ref.copy()
    flip[:, 3:7] = quat.quat_from_axis_angle(Y, np.linspace(0, -2 * np.pi, n_t), np)
    assert metrics.is_flip_clip(flip)


def _tracking_inputs(done_at: dict[int, int], n: int = 3, n_t: int = 50):
    done = np.zeros((n, n_t))
    for ep, step in done_at.items():
        done[ep, step] = 1.0
    qpos = np.zeros((n, n_t, 9))
    qpos[..., 3] = 1.0
    return np.ones((n, n_t)), done, qpos


def test_tracking_cause_rates_separate_finish_from_failure():
    # ep0 reaches the clip end (done at the last step, no cause flag), ep1 falls, ep2 loses tracking.
    reward, done, qpos = _tracking_inputs({0: 49, 1: 20, 2: 30})
    fell = np.zeros_like(done)
    failed = np.zeros_like(done)
    fell[1, 20] = 1.0
    failed[2, 30] = 1.0
    out = metrics.summarize_tracking(reward, done, qpos, qpos, DT, fell, failed)
    assert out["terminated_early_rate"] == 1.0  # old key still counts finishing as termination
    assert out["completion_rate"] == pytest.approx(1 / 3)
    assert out["fall_rate"] == pytest.approx(1 / 3)
    assert out["tracking_fail_rate"] == pytest.approx(1 / 3)
    assert out["survival_time_s_mean"] == pytest.approx((50 + 21 + 31) / 3 * DT)


def test_tracking_fell_and_failed_same_step_counts_as_fall_only():
    reward, done, qpos = _tracking_inputs({0: 10}, n=1)
    flag = np.zeros_like(done)
    flag[0, 10] = 1.0
    out = metrics.summarize_tracking(reward, done, qpos, qpos, DT, flag, flag)
    assert out["fall_rate"] == 1.0 and out["tracking_fail_rate"] == 0.0 and out["completion_rate"] == 0.0


def test_tracking_cause_flags_after_first_done_are_ignored():
    reward, done, qpos = _tracking_inputs({0: 49}, n=1)
    fell = np.zeros_like(done)
    fell[0, 49] = 0.0
    out = metrics.summarize_tracking(reward, done, qpos, qpos, DT, fell, np.zeros_like(done))
    assert out["completion_rate"] == 1.0 and out["fall_rate"] == 0.0


def test_eval_mode_starts_mimic_at_clip_start():
    pytest.importorskip("jax")
    pytest.importorskip("mujoco_playground")
    from humanoid_mimic_rl import tasks
    from humanoid_mimic_rl.envs import mimic

    assert tasks.mimic_overrides(False, None) == {}
    assert tasks.mimic_overrides(True, None) == {"rsi_start_prob": 1.0}
    assert tasks.mimic_overrides(False, "height_tilt") == {"termination": "height_tilt"}
    assert tasks.mimic_overrides(False, "height_tilt", 1.5) == {"termination": "height_tilt", "max_tilt_err": 1.5}
    assert tasks.mimic_overrides(False, None, None, (0.3, 0.75, 0.5)) == {
        "rsi_focus_start_s": 0.3, "rsi_focus_end_s": 0.75, "rsi_focus_prob": 0.5}
    assert tasks.mimic_overrides(True, None, None, (0.3, 0.75, 0.5)) == {"rsi_start_prob": 1.0}  # eval: t = 0
    assert mimic.default_config().rsi_start_prob < 1.0  # training keeps RSI
