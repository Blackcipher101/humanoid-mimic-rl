import dataclasses

import numpy as np
import pytest

from humanoid_mimic_rl import clip_builder, keyframes as kf, quat, t1_model


def test_pchip_passes_through_knots_and_is_monotone():
    x = np.array([0.0, 0.3, 0.65, 0.9, 1.4])
    y = np.array([0.0, 0.0, -1.0, -3.0, -6.3])
    xq = np.linspace(0.0, 1.4, 500)
    yq = kf.pchip_eval(x, y, xq)
    np.testing.assert_allclose(kf.pchip_eval(x, y, x), y, atol=1e-12)
    assert np.all(np.diff(yq) <= 1e-12)  # monotone data -> monotone interpolant


def test_pchip_no_overshoot_beyond_local_extrema():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([[0.0], [2.34], [2.34], [0.0]])  # e.g. a knee held at its limit
    yq = kf.pchip_eval(x, y, np.linspace(0, 3, 301))
    assert yq.max() <= 2.34 + 1e-12 and yq.min() >= -1e-12


def test_pchip_two_knots_is_linear():
    yq = kf.pchip_eval(np.array([0.0, 2.0]), np.array([1.0, 3.0]), np.array([0.5, 1.0]))
    np.testing.assert_allclose(yq, [1.5, 2.0])


def test_backflip_keyframes_valid_for_t1():
    keys = kf.t1_backflip_keyframes()
    kf.validate_keyframes(keys, t1_model.T1_JOINT_NAMES, t1_model.joint_ranges())
    assert keys[-1].pitch_rad == pytest.approx(-2 * np.pi)
    flight = kf.flight_phase(keys)
    assert flight.takeoff_s == pytest.approx(0.90) and flight.landing_s == pytest.approx(1.42)


def test_interpolated_backflip_within_joint_limits():
    keys = kf.t1_backflip_keyframes()
    traj = kf.interpolate_keyframes(keys, t1_model.T1_JOINT_NAMES, t1_model.home_pose(), 100.0)
    lo, hi = t1_model.joint_ranges().T
    assert np.all(traj.joint_pos_rad >= lo - 1e-9) and np.all(traj.joint_pos_rad <= hi + 1e-9)
    assert traj.times_s[-1] == pytest.approx(keys[-1].time_s)
    np.testing.assert_allclose(traj.joint_pos_rad[0], t1_model.home_pose())
    airborne = ~traj.in_contact
    assert traj.times_s[airborne].min() > 0.90 and traj.times_s[airborne].max() < 1.42


@pytest.mark.parametrize(
    "keys, msg",
    [
        ([kf.Keyframe(0.0, {}, 0.0, True)], "two"),
        ([kf.Keyframe(0.1, {}, 0.0, True), kf.Keyframe(0.2, {}, 0.0, True)], "t = 0"),
        ([kf.Keyframe(0.0, {}, 0.0, True), kf.Keyframe(0.0, {}, 0.0, True)], "increasing"),
        ([kf.Keyframe(0.0, {"Nope": 0.0}, 0.0, True), kf.Keyframe(1.0, {}, 0.0, True)], "unknown"),
        ([kf.Keyframe(0.0, {"Left_Knee_Pitch": 3.0}, 0.0, True), kf.Keyframe(1.0, {}, 0.0, True)], "outside"),
    ],
)
def test_validate_rejects(keys, msg):
    with pytest.raises(ValueError, match=msg):
        kf.validate_keyframes(keys, t1_model.T1_JOINT_NAMES, t1_model.joint_ranges())


def test_flight_phase_rejects_two_flights():
    keys = [kf.Keyframe(t, {}, 0.0, c) for t, c in [(0, True), (1, False), (2, True), (3, False), (4, True)]]
    with pytest.raises(ValueError, match="more than one"):
        kf.flight_phase(keys)


def test_contact_at_segments():
    keys = [kf.Keyframe(t, {}, 0.0, c) for t, c in [(0.0, True), (1.0, True), (2.0, False), (3.0, True)]]
    c = kf.contact_at(keys, np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]))
    np.testing.assert_array_equal(c, [True, True, False, False, False, True])


def test_ballistic_takeoff_velocity():
    # Same height in 0.5 s: v0 = g T / 2.
    assert kf.ballistic_takeoff_velocity(0.6, 0.6, 0.5, 9.81) == pytest.approx(9.81 * 0.25)
    with pytest.raises(ValueError):
        kf.ballistic_takeoff_velocity(0.0, 0.0, 0.0, 9.81)


@dataclasses.dataclass
class ToyKinematics:
    """Rigid body: CoM 0.05 m along body z from the root, one foot 0.6 m along -z."""

    site_names: tuple[str, ...] = ("foot",)

    def forward(self, qpos: np.ndarray) -> clip_builder.FKResult:
        root, q = qpos[0:3], qpos[3:7]
        com = root + quat.quat_rotate(q, np.array([0.0, 0.0, 0.05]), np)
        foot = root + quat.quat_rotate(q, np.array([0.0, 0.0, -0.6]), np)
        return clip_builder.FKResult(com=com, foot_points=foot[None, :], site_pos=foot[None, :])


def _toy_traj() -> tuple[kf.KeyframeTrajectory, kf.FlightPhase]:
    keys = [
        kf.Keyframe(0.0, {}, 0.0, True),
        kf.Keyframe(0.4, {}, 0.0, True),
        kf.Keyframe(0.65, {}, -np.pi, False),
        kf.Keyframe(0.9, {}, -2 * np.pi, True),
        kf.Keyframe(1.2, {}, -2 * np.pi, True),
    ]
    traj = kf.interpolate_keyframes(keys, ("j",), np.zeros(1), 100.0)
    return traj, kf.flight_phase(keys)


def test_build_clip_ballistic_com_and_ground_contact():
    traj, flight = _toy_traj()
    diag, clip = clip_builder.build_clip(traj, flight, ToyKinematics(), ("j",), -0.2, "lic", "src")
    kin = ToyKinematics()
    fk = [kin.forward(q) for q in clip.qpos]
    com = np.array([f.com for f in fk])
    foot_z = np.array([f.foot_points[0, 2] for f in fk])
    stance = traj.in_contact
    np.testing.assert_allclose(foot_z[stance], 0.0, atol=1e-12)
    # Constant vertical acceleration -g during flight.
    t = traj.times_s
    fl = (t > flight.takeoff_s + 0.015) & (t < flight.landing_s - 0.015)
    acc = np.gradient(np.gradient(com[:, 2], 0.01), 0.01)
    np.testing.assert_allclose(acc[fl], -clip_builder.GRAVITY_MPS2, atol=1e-6)
    assert diag.flight_time_s == pytest.approx(0.5)
    assert diag.takeoff_vz_mps == pytest.approx(clip_builder.GRAVITY_MPS2 * 0.25, rel=1e-6)
    assert diag.landing_offset_x_m == -0.2
    np.testing.assert_allclose(fk[-1].foot_points[0, 0], -0.2, atol=1e-12)
    assert clip.licence == "lic"


def test_build_clip_auto_landing_offset_matches_stance_velocity():
    traj, flight = _toy_traj()
    diag, _ = clip_builder.build_clip(traj, flight, ToyKinematics(), ("j",), None, "lic", "src")
    # Toy stance motion is stationary at take-off -> no horizontal travel.
    assert diag.landing_offset_x_m == pytest.approx(0.0, abs=1e-9)
