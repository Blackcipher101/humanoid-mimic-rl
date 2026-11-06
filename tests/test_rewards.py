import numpy as np
import pytest

from humanoid_mimic_rl import quat, rewards

Y = np.array([0.0, 1.0, 0.0])


def test_exp_kernel_and_ramp():
    assert rewards.exp_kernel(0.0, 2.0, np) == 1.0
    assert rewards.exp_kernel(1.0, 2.0, np) == pytest.approx(np.exp(-2.0))
    np.testing.assert_allclose(rewards.ramp_reward(np.array([-1.0, 1.0, 5.0]), 2.0, np), [0.0, 0.5, 1.0])


def test_time_window_is_half_open():
    t = np.array([0.0, 0.5, 1.0, 1.5])
    np.testing.assert_array_equal(rewards.time_window(t, 0.5, 1.5, np), [0, 1, 1, 0])
    assert rewards.time_window(10.0, 1.0, np.inf, np) == 1.0


def test_upright_reward_range_and_monotonic():
    up = np.linspace(-1.0, 1.0, 11)
    r = rewards.upright_reward(up, np)
    assert r[0] == 0.0 and r[-1] == 1.0
    assert np.all(np.diff(r) > 0)
    assert rewards.upright_reward(1.5, np) == 1.0  # clipped


def test_height_reward_saturates():
    np.testing.assert_allclose(rewards.height_reward(np.array([0.0, 0.33, 1.0]), 0.66, np), [0, 0.5, 1])


def test_tall_upright_removes_sitting_plateau():
    # sitting upright (torso vertical, half height) must score far below standing, unlike upright_reward alone
    sit = rewards.tall_upright_reward(1.0, 0.33, 0.66, np)
    stand = rewards.tall_upright_reward(1.0, 0.66, 0.66, np)
    lying = rewards.tall_upright_reward(0.0, 0.15, 0.66, np)
    assert stand == 1.0 and sit == 0.25 and lying < sit
    assert rewards.upright_reward(1.0, np) == 1.0  # the old term paid sitting fully


def test_feet_under_torso_reward():
    torso = np.array([0.0, 0.0])
    under = np.array([[0.05, 0.1], [-0.05, -0.1]])
    far = np.array([[0.6, 0.1], [0.5, -0.1]])
    assert rewards.feet_under_torso_reward(torso, under, 0.25, np) == 1.0
    r_far = rewards.feet_under_torso_reward(torso, far, 0.25, np)
    np.testing.assert_allclose(r_far, np.exp(-(0.55 / 0.25) ** 2))
    batch = rewards.feet_under_torso_reward(np.zeros((3, 2)), np.stack([under, far, under]), 0.25, np)
    assert batch.shape == (3,)


def test_standing_gate():
    up = np.array([0.95, 0.95, 0.5])
    h = np.array([0.7, 0.3, 0.7])
    np.testing.assert_array_equal(rewards.standing_gate(up, h, 0.9, 0.6, np), [1, 0, 0])


def test_posture_reward_weights():
    q = np.array([0.1, 1.0])
    target = np.zeros(2)
    assert rewards.posture_reward(q, target, np.array([1.0, 0.0]), 1.0, np) == pytest.approx(np.exp(-0.01))


def test_costs_non_negative_and_zero_at_rest():
    a = np.zeros(3)
    assert rewards.action_rate_cost(a, a, a, np) == 0.0
    assert rewards.action_rate_cost(np.ones(3), a, a, np) == pytest.approx(3.0 + 3.0)
    lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    assert rewards.joint_limit_cost(np.array([0.0, 0.5]), lo, hi, np) == 0.0
    assert rewards.joint_limit_cost(np.array([-1.5, 1.25]), lo, hi, np) == pytest.approx(0.75)
    assert rewards.joint_velocity_excess_cost(np.array([3.0, -12.0]), 10.0, np) == pytest.approx(4.0)
    assert rewards.torque_cost(np.array([3.0, 4.0]), np) == pytest.approx(25.0)


def test_backward_pitch_rate_sign():
    # Backflip = negative rotation about body y.
    assert rewards.backward_pitch_rate(np.array([0.0, -5.0, 0.0])) == 5.0


def test_pitch_quat_backwards_moves_head_to_minus_x():
    q = rewards.pitch_quat(-np.pi / 2, np)
    z_world = quat.quat_rotate(q, np.array([0.0, 0.0, 1.0]), np)
    np.testing.assert_allclose(z_world, [-1.0, 0.0, 0.0], atol=1e-12)


def test_orientation_tracking_wraps_full_turn():
    q = rewards.pitch_quat(0.0, np)
    ref = rewards.pitch_quat(-2 * np.pi, np)
    assert rewards.orientation_tracking_reward(q, ref, 1.0, np) == pytest.approx(1.0)
    ref90 = rewards.pitch_quat(-np.pi / 2, np)
    assert rewards.orientation_tracking_reward(q, ref90, 1.0, np) == pytest.approx(np.exp(-(np.pi / 2) ** 2))


def test_landing_reward_requires_contact_and_uprightness():
    up = np.array([1.0, 1.0, 0.5])
    h = np.array([0.6, 0.6, 0.6])
    contact = np.array([[True, False], [False, False], [True, True]])
    np.testing.assert_allclose(rewards.landing_reward(up, h, contact, 0.8, 0.5, np), [1.0, 0.0, 0.0])


def _deepmimic_inputs():
    q = np.array([0.1, -0.2, 0.3])
    rq = quat.quat_from_axis_angle(Y, 0.4, np)
    pos = np.array([0.0, 0.0, 0.7])
    ee = np.array([[0.1, 0.1, 0.0], [0.1, -0.1, 0.0]])
    return q, rq, pos, ee


def test_deepmimic_perfect_tracking_gives_one():
    q, rq, pos, ee = _deepmimic_inputs()
    gains = {"pose": 2.0, "velocity": 0.1, "end_effector": 40.0, "root": 10.0}
    terms = rewards.deepmimic_terms(q, q, q, q, rq, rq, pos, pos, ee, ee, gains, np)
    for v in terms.values():
        assert v == pytest.approx(1.0)


def test_deepmimic_individual_terms():
    q, rq, pos, ee = _deepmimic_inputs()
    gains = {"pose": 2.0, "velocity": 0.1, "end_effector": 40.0, "root": 10.0}
    rq_off = quat.quat_mul(rq, quat.quat_from_axis_angle(Y, 0.1, np), np)
    t = rewards.deepmimic_terms(
        q + np.array([0.1, 0.0, 0.0]), q, q + 1.0, q, rq_off, rq,
        pos + np.array([0.1, 0.0, 0.0]), pos, ee + np.array([0.1, 0.0, 0.0]), ee, gains, np,
    )
    assert t["pose"] == pytest.approx(np.exp(-2.0 * (0.01 + 0.01)))
    assert t["velocity"] == pytest.approx(np.exp(-0.1 * 3.0))
    # End effectors are measured relative to the root: shifting both by the
    # same offset is not an error.
    assert t["end_effector"] == pytest.approx(1.0)
    assert t["root"] == pytest.approx(np.exp(-10.0 * 0.01))


def test_tracking_failed_thresholds():
    rq = quat.quat_from_axis_angle(Y, 0.0, np)
    far = quat.quat_from_axis_angle(Y, 1.3, np)
    p = np.zeros(3)
    assert not rewards.tracking_failed(p, p + 0.5, rq, rq, 1.0, 1.2, np)
    assert rewards.tracking_failed(p, p + 1.0, rq, rq, 1.0, 1.2, np)
    assert rewards.tracking_failed(p, p, rq, far, 1.0, 1.2, np)


def test_home_range_targets():
    home = np.array([0.4, -0.2])
    lo, hi = np.array([0.0, -0.87]), np.array([2.34, 0.35])
    np.testing.assert_allclose(rewards.home_range_targets(np.zeros(2), home, lo, hi, np), home)
    np.testing.assert_allclose(rewards.home_range_targets(np.ones(2), home, lo, hi, np), hi)
    np.testing.assert_allclose(rewards.home_range_targets(-np.ones(2), home, lo, hi, np), lo)
    np.testing.assert_allclose(rewards.home_range_targets(np.array([0.5, -0.5]), home, lo, hi, np),
                               [0.4 + 0.5 * 1.94, -0.2 - 0.5 * 0.67])
    np.testing.assert_allclose(rewards.home_range_targets(np.array([3.0, -3.0]), home, lo, hi, np), [2.34, -0.87])
