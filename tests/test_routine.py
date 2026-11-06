"""Routine milestones, per-milestone bonuses, soft surrogate, metrics and the routine clip (numpy only)."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from humanoid_mimic_rl import metrics, quat, reference, routine

DT = 0.02
WIN = routine.RoutineWindows(0.2, 1.0, 1.5, 2.5, 0, 1.0, 1.0)
THR = routine.RoutineThresholds()
END_S = 3.0


def _quat(yaw, pitch):
    qz = quat.quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw, np)
    qy = quat.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), pitch, np)
    return quat.quat_mul(qz, qy, np)


def _ramp(t, t0, t1, total):
    return total * float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))


def _episode(flip_deg=340.0, land=True, kick_deg=600.0, foot_z=1.0, fall_at=None, stand=True):
    """Run ``routine.update`` + bonuses over a synthetic episode; returns dict of totals and final flags."""
    state = routine.initial_state(_quat(0.0, 0.0), np)
    bonus_steps, flags = [], None
    fell_now = False
    n = int(round(END_S / DT))
    for k in range(1, n + 1):
        t = k * DT
        pitch = -math.radians(_ramp(t, 0.2, 0.9, flip_deg))  # backward = negative body-y rotation
        yaw = math.radians(_ramp(t, 1.5, 2.4, kick_deg))
        q = _quat(yaw, pitch)
        up_z = float(1 - 2 * (q[1] ** 2 + q[2] ** 2))
        landed_phase = t >= 0.95
        contact = np.array([land and landed_phase] * 2 or [False, False])
        fell_now = fall_at is not None and t >= fall_at
        z = foot_z if 1.7 <= t <= 2.0 else 0.04
        up = up_z if (stand or t < 2.8) else 0.3
        old = state
        state, flags = routine.update(state, t, q, up, contact, z, np.bool_(fell_now), THR, WIN, np)
        finished = np.bool_(k == n)
        ok = routine.normal_finish(finished, np.bool_(fell_now), np.bool_(False), np.bool_(False))
        b = routine.bonus_reward(old, flags, t, DT, ok, 1.0, 10.0, 100.0, WIN, np)
        bonus_steps.append(float(b))
    return {"flags": flags, "bonus": np.array(bonus_steps)}


def test_rotation_accumulates_through_euler_wrap():
    pitch = -np.linspace(0.0, 3 * 2 * np.pi, 400)  # three backward turns: Euler angle wraps six times
    q = np.array([_quat(0.0, p) for p in pitch])
    d_pitch, d_yaw = routine.rotation_increments(q[:-1], q[1:], np)
    assert -np.sum(d_pitch) == pytest.approx(3 * 2 * np.pi, rel=1e-6)
    assert np.allclose(d_yaw, 0.0, atol=1e-9)
    yaw = np.linspace(0.0, 2 * 2 * np.pi, 400)
    qy = np.array([_quat(a, 0.0) for a in yaw])
    assert np.sum(routine.rotation_increments(qy[:-1], qy[1:], np)[1]) == pytest.approx(2 * 2 * np.pi, rel=1e-6)


def test_rotation_increment_ignores_nonfinite():
    q = _quat(0.0, 0.0)
    d_pitch, d_yaw = routine.rotation_increments(q, np.full(4, np.nan), np)
    assert d_pitch == 0.0 and d_yaw == 0.0


@pytest.mark.parametrize(
    "kwargs, flip, kick, routine_done",
    [
        (dict(), True, True, True),  # both
        (dict(kick_deg=100.0), True, False, False),  # flip only
        (dict(flip_deg=200.0), False, True, False),  # kick only
        (dict(flip_deg=200.0, kick_deg=100.0), False, False, False),
        (dict(land=False), False, True, False),  # rotated but never landed
        (dict(foot_z=0.3), True, False, False),  # kick foot never reaches 0.6 x peak
        (dict(stand=False), True, True, False),  # not upright at the end
    ],
)
def test_milestone_truth_table(kwargs, flip, kick, routine_done):
    out = _episode(**kwargs)["flags"]
    assert (bool(out["flip_done"]), bool(out["kick_done"]), bool(out["routine_done"])) == (flip, kick, routine_done)


def test_both_but_fell_is_not_routine_done():
    out = _episode(fall_at=2.9)["flags"]
    assert bool(out["flip_done"]) and bool(out["kick_done"]) and not bool(out["routine_done"])


def test_bonuses_paid_once_at_their_step_and_stand_only_on_final_step():
    b = _episode()["bonus"]
    steps = np.flatnonzero(b)
    assert b.sum() == pytest.approx(1.0 + 10.0 + 100.0)
    assert len(steps) == 3
    flip_step, kick_step, stand_step = steps
    assert b[flip_step] == 1.0 and b[kick_step] == 10.0 and b[stand_step] == 100.0
    assert stand_step == len(b) - 1
    assert (kick_step + 1) * DT > WIN.kick_end_s  # paid just after the kick window closes
    assert (kick_step + 1) * DT < WIN.kick_end_s + 2 * DT


def test_missed_milestones_pay_nothing_and_fall_pays_no_stand_bonus():
    b = _episode(kick_deg=100.0)["bonus"]
    assert b.sum() == pytest.approx(101.0)  # flip + standing, no kick bonus
    b = _episode(fall_at=2.9)["bonus"]
    assert b[-1] == 0.0 and b.sum() == pytest.approx(11.0)


def test_bonus_is_finite_and_zero_off_the_paid_steps():
    b = _episode()["bonus"]
    assert np.all(np.isfinite(b)) and np.count_nonzero(b) == 3


def test_gate_zeroes_reward_after_missed_window():
    g = routine.gate
    assert g(0.5, False, False, THR, WIN, np) == 1.0
    assert g(1.6, False, False, THR, WIN, np) == 0.0  # flip window + grace closed unmet
    assert g(1.6, True, False, THR, WIN, np) == 1.0
    assert g(2.6, True, False, THR, WIN, np) == 0.0


def test_soft_surrogate_is_finite_monotone_and_bounded():
    prog = routine.ReferenceProgress(np.zeros(3), np.zeros(3), np.zeros(3, dtype=bool), np.zeros(3))
    state = routine.soft_initial_state(_quat(0.0, 0.0), 0.0, prog, 0, THR, WIN, np)
    total_flip = total_kick = 0.0
    n = int(round(END_S / DT))
    for k in range(1, n + 1):
        t = k * DT
        q = _quat(math.radians(_ramp(t, 1.5, 2.4, 600.0)), -math.radians(_ramp(t, 0.2, 0.9, 340.0)))
        feet = np.array([0.02, 0.02]) if t >= 0.95 else np.array([0.5, 0.5])
        state, d_flip, d_kick = routine.soft_update(state, t, q, 1.0, feet, 1.0 if 1.7 <= t <= 2.0 else 0.04, THR, WIN, np)
        assert d_flip >= 0.0 and d_kick >= 0.0 and np.isfinite(d_flip) and np.isfinite(d_kick)
        total_flip += d_flip
        total_kick += d_kick
    assert 0.8 < total_flip <= 1.0 and 0.8 < total_kick <= 1.0
    assert routine.soft_stand(1.0, True, False, THR, np) > 0.85
    assert routine.soft_stand(1.0, False, False, THR, np) == 0.0
    assert routine.soft_stand(1.0, True, True, THR, np) == 0.0
    state, d_flip, d_kick = routine.soft_update(state, 1.0, np.full(4, np.nan), 1.0, np.zeros(2), 0.0, THR, WIN, np)
    assert np.isfinite(d_flip) and np.isfinite(d_kick)


def test_soft_update_handles_extreme_inputs_without_overflow():
    prog = routine.ReferenceProgress(np.zeros(3), np.zeros(3), np.zeros(3, dtype=bool), np.zeros(3))
    state = routine.soft_initial_state(_quat(0.0, 0.0), 0.0, prog, 0, THR, WIN, np)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        _, d_flip, d_kick = routine.soft_update(state, 1.0, _quat(0.0, 0.0), -1e6, np.array([1e6, 1e6]), 1e6, THR, WIN, np)
    assert np.isfinite(d_flip) and np.isfinite(d_kick)


def test_threshold_and_window_validation():
    routine.RoutineThresholds().validate()
    for bad in (dict(flip_min_rotation_rad=0.0), dict(kick_min_height_frac=1.5), dict(land_min_up_z=1.0),
                dict(flip_bonus_fraction=-0.1), dict(soft_height_scale_m=0.0), dict(flip_landing_grace_s=float("nan"))):
        with pytest.raises(ValueError):
            routine.RoutineThresholds(**bad).validate()
    with pytest.raises(ValueError, match="before the kick"):
        routine.RoutineWindows(2.0, 3.0, 1.0, 2.0, 0, 1.0, 1.0).validate()
    with pytest.raises(ValueError, match="routine"):
        routine.windows_from_report({}, ("left_foot",))


def test_reference_progress_and_credit():
    n = 160
    pitch = -np.radians(np.clip((np.arange(n) * DT - 0.2) / 0.7, 0, 1) * 340.0)
    qpos = np.zeros((n, 7))
    qpos[:, 3:7] = np.array([_quat(0.0, p) for p in pitch])
    prog = routine.reference_progress(qpos, np.full(n, 0.5), 1.0 / DT, THR, WIN)
    assert prog.flip_rot_rad[-1] == pytest.approx(math.radians(340.0), rel=1e-6)
    late = routine.credit_at(prog, np.int32(n - 1), (n - 1) * DT, THR, WIN, np)
    assert bool(late["flip_rot_hit"]) and bool(late["flip_landed"])
    early = routine.credit_at(prog, np.int32(5), 5 * DT, THR, WIN, np)
    assert not bool(early["flip_rot_hit"])


def test_summarize_routine_rates_use_edge_flags_before_termination():
    done = np.zeros((3, 5))
    done[1, 2] = 1  # episode 1 terminates at step 2
    flip = np.zeros((3, 5))
    flip[0, 1] = flip[1, 1] = 1
    kick = np.zeros((3, 5))
    kick[0, 3] = 1
    kick[1, 4] = 1  # after termination: ignored
    success = np.zeros((3, 5))
    success[0, 4] = 1
    out = metrics.summarize_routine(done, flip, kick, success)
    assert out == {"flip_rate": pytest.approx(2 / 3), "kick_rate": pytest.approx(1 / 3),
                   "routine_success_rate": pytest.approx(1 / 3)}


# ----------------------------------------------------------------------
# The routine clip (needs the generated data and the G1 model files).
# ----------------------------------------------------------------------

_CLIP = Path(__file__).resolve().parent.parent / "data" / "generated" / "g1_routine_backflip_spinkick.npz"


@pytest.mark.skipif(not _CLIP.is_file(), reason="routine clip not generated (scripts/make_g1_routine.py)")
def test_routine_clip_continuity_and_report():
    clip = reference.load_clip(_CLIP)
    report = json.loads(_CLIP.with_suffix(".report.json").read_text(encoding="utf-8"))
    cont = report["continuity"]
    assert cont["max_root_step_cm_transition"] <= 2.0
    assert cont["max_root_step_cm_final_stand"] <= 2.0
    assert cont["max_planted_foot_slip_cm"] < 1.0
    assert report["torque_scale"] == 1.0 and report["passed"] is False
    assert "may not resell" in report["licence_statement"] and "redistribution_note" in report
    assert {"87", "88"} == {str(s["trial"]["subject"]) for s in report["sources"].values()}
    assert all(s["source_files"]["amc"]["url"].startswith("http://mocap.cs.cmu.edu") for s in report["sources"].values())
    for key in ("check", "segments_s", "routine"):
        assert key in report
    assert report["check"]["torque_ratio_max"] > 1.0
    win = routine.windows_from_report(report, clip.site_names)
    assert win.flip_end_s < win.kick_start_s + 1.0 and win.kick_end_s < clip.duration_s
    # qvel consistent with qpos (finite differences, 50 Hz) and the reference satisfies its own milestones
    assert clip.fps_hz == 50.0
    assert np.allclose(clip.qvel, reference.finite_difference_qvel(clip.qpos, clip.fps_hz))
    prog = routine.reference_progress(clip.qpos, clip.site_pos[:, win.kick_foot_site, 2], clip.fps_hz, THR, win)
    assert prog.flip_rot_rad.max() >= THR.flip_min_rotation_rad
    assert prog.kick_yaw_rad.max() >= THR.kick_min_rotation_rad and prog.kick_foot_hit[-1]


@pytest.mark.skipif(not _CLIP.is_file() or "G1_XML_DIR" not in os.environ, reason="needs the clip and G1 model files")
def test_routine_transition_root_height_continuous():
    clip = reference.load_clip(_CLIP)
    report = json.loads(_CLIP.with_suffix(".report.json").read_text(encoding="utf-8"))
    t0, t1 = report["segments_s"]["transition"]
    k0, k1 = int(round(t0 * clip.fps_hz)), int(round(t1 * clip.fps_hz))
    assert np.max(np.abs(np.diff(clip.qpos[k0 - 2 : k1 + 3, 2]))) < 0.02
    assert np.max(np.linalg.norm(np.diff(clip.qpos[k0 - 2 : k1 + 3, :2], axis=0), axis=1)) < 0.02
