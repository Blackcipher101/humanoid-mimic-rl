"""Offline evaluation metrics computed from batched rollouts (numpy only).

All inputs are ``(N, T, ...)`` arrays for ``N`` episodes of ``T`` control
steps recorded **after** each step. ``done[n, t]`` marks the step at which
episode ``n`` terminated; data after the first ``done`` is ignored.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from humanoid_mimic_rl import quat

TWO_PI = 2.0 * np.pi


def alive_mask(done: np.ndarray) -> np.ndarray:
    """``(N, T)`` bool: True up to and including the first terminal step."""
    d = done.astype(bool)
    terminated_before = np.cumsum(d, axis=1) - d  # number of dones strictly before t
    return terminated_before == 0


def terminated_early(done: np.ndarray) -> np.ndarray:
    """``(N,)`` bool: the episode hit a terminal state before the time limit."""
    return np.any(done.astype(bool), axis=1)


def episode_returns(reward: np.ndarray, done: np.ndarray) -> np.ndarray:
    return np.sum(reward * alive_mask(done), axis=1)


def episode_lengths(done: np.ndarray) -> np.ndarray:
    return np.sum(alive_mask(done), axis=1)


def last_alive_index(done: np.ndarray) -> np.ndarray:
    return episode_lengths(done) - 1


def cumulative_pitch_rad(root_quat: np.ndarray) -> np.ndarray:
    """Accumulated rotation (rad) about the body y-axis, starting at 0.

    Uses the body-frame relative rotation between consecutive samples,
    ``log(q_t^-1 q_t+1)``, so it does not wrap at +/-pi and counts full turns.
    Valid while the per-sample rotation stays below pi (12 rad/s at 50 Hz is
    0.24 rad). Negative values = backward rotation.
    """
    rel = quat.quat_mul(quat.quat_conj(root_quat[..., :-1, :], np), root_quat[..., 1:, :], np)
    d_pitch = quat.quat_to_rotvec(rel, np)[..., 1]
    zeros = np.zeros(root_quat.shape[:-2] + (1,))
    return np.concatenate([zeros, np.cumsum(d_pitch, axis=-1)], axis=-1)


def _masked_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-episode mean over masked steps; NaN for episodes with no valid step."""
    count = np.sum(mask, axis=1)
    total = np.sum(np.where(mask, x, 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _final(x: np.ndarray, done: np.ndarray) -> np.ndarray:
    idx = last_alive_index(done)
    return np.take_along_axis(x, idx.reshape((-1, 1) + (1,) * (x.ndim - 2)), axis=1)[:, 0]


@dataclasses.dataclass(frozen=True)
class FallCriteria:
    """A torso tilted beyond ``acos(min_up_z)`` or lower than ``min_height_m`` has fallen."""

    min_up_z: float = 0.5  # 60 deg tilt
    min_height_m: float = 0.4


def fallen(up_z: np.ndarray, height_m: np.ndarray, crit: FallCriteria) -> np.ndarray:
    return (up_z < crit.min_up_z) | (height_m < crit.min_height_m)


def summarize_common(reward: np.ndarray, done: np.ndarray, ctrl_dt_s: float) -> dict[str, float]:
    returns = episode_returns(reward, done)
    lengths = episode_lengths(done)
    return {
        "num_episodes": int(reward.shape[0]),
        "episode_return_mean": float(np.mean(returns)),
        "episode_return_std": float(np.std(returns)),
        "episode_length_s_mean": float(np.mean(lengths) * ctrl_dt_s),
        "terminated_early_rate": float(np.mean(terminated_early(done))),
    }


def summarize_walk(
    reward: np.ndarray,
    done: np.ndarray,
    local_linvel_mps: np.ndarray,
    command: np.ndarray,
    up_z: np.ndarray,
    height_m: np.ndarray,
    ctrl_dt_s: float,
    crit: FallCriteria,
) -> dict[str, float]:
    """Velocity tracking. ``command`` = (vx m/s, vy m/s, yaw rate rad/s) in the body frame.

    Success = completed the episode without terminating and without meeting
    the fall criteria at any step.
    """
    mask = alive_mask(done)
    fell = np.any(mask & fallen(up_z, height_m, crit), axis=1) | terminated_early(done)
    vx_err = np.abs(local_linvel_mps[..., 0] - command[..., 0])
    vxy_err = np.linalg.norm(local_linvel_mps[..., :2] - command[..., :2], axis=-1)
    out = summarize_common(reward, done, ctrl_dt_s)
    out.update(
        fall_rate=float(np.mean(fell)),
        success_rate=float(np.mean(~fell)),
        forward_speed_abs_error_mps=float(np.nanmean(_masked_mean(vx_err, mask))),
        planar_velocity_error_mps=float(np.nanmean(_masked_mean(vxy_err, mask))),
        forward_speed_mean_mps=float(np.nanmean(_masked_mean(local_linvel_mps[..., 0], mask))),
        commanded_forward_speed_mps=float(np.mean(command[:, 0, 0])),
    )
    return out


def first_sustained(flag: np.ndarray, hold_steps: int) -> np.ndarray:
    """Index of the first step at which ``flag`` has been True for ``hold_steps`` consecutive steps; -1 if never."""
    n, t = flag.shape
    out = np.full(n, -1)
    run = np.zeros(n, dtype=int)
    for k in range(t):
        run = np.where(flag[:, k], run + 1, 0)
        newly = (run >= hold_steps) & (out < 0)
        out = np.where(newly, k, out)
    return out


def summarize_getup(
    reward: np.ndarray,
    done: np.ndarray,
    standing: np.ndarray,
    ctrl_dt_s: float,
    hold_s: float,
) -> dict[str, float]:
    """Get-up success = ``standing`` held for the final ``hold_s`` of the episode.

    ``standing`` is the environment's upright-and-high gate per step.
    ``fall_rate`` = fraction of episodes that stood (for ``hold_s``) at some
    point but were not standing at the end.
    """
    hold_steps = max(1, int(round(hold_s / ctrl_dt_s)))
    mask = alive_mask(done)
    ok = standing.astype(bool) & mask
    stood_at = first_sustained(ok, hold_steps)
    final_ok = np.all(ok[:, -hold_steps:], axis=1) & ~terminated_early(done)
    stood = stood_at >= 0
    out = summarize_common(reward, done, ctrl_dt_s)
    times = (stood_at[stood] - hold_steps + 1) * ctrl_dt_s
    out.update(
        success_rate=float(np.mean(final_ok)),
        stood_up_rate=float(np.mean(stood)),
        fall_rate=float(np.mean(stood & ~final_ok)),
        time_to_stand_s_median=float(np.median(times)) if times.size else float("nan"),
    )
    return out


def summarize_flip(
    reward: np.ndarray,
    done: np.ndarray,
    root_quat: np.ndarray,
    up_z: np.ndarray,
    height_m: np.ndarray,
    feet_contact: np.ndarray,
    ctrl_dt_s: float,
    rotation_tolerance_rad: float,
    land_min_up_z: float,
    land_min_height_m: float,
) -> dict[str, float]:
    """Backflip metrics.

    * ``max_backward_rotation_rad``: max over time of the accumulated backward
      pitch (2*pi = one full backflip), mean over episodes;
    * landing success: rotated at least ``2*pi - rotation_tolerance_rad``,
      never terminated (no head/trunk ground contact, no tracking failure),
      and at the final step upright (``up_z >= land_min_up_z``), torso height
      ``>= land_min_height_m`` and at least one foot on the ground;
    * ``fall_rate``: terminated early or not upright/high at the final step.
    """
    mask = alive_mask(done)
    pitch = cumulative_pitch_rad(root_quat)
    backward = np.where(mask, -pitch, -np.inf)
    max_back = np.max(backward, axis=1)
    early = terminated_early(done)
    final_up = _final(up_z, done) >= land_min_up_z
    final_high = _final(height_m, done) >= land_min_height_m
    final_contact = np.any(_final(feet_contact, done), axis=-1)
    upright_end = final_up & final_high & ~early
    rotated = max_back >= TWO_PI - rotation_tolerance_rad
    landed = rotated & upright_end & final_contact
    out = summarize_common(reward, done, ctrl_dt_s)
    out.update(
        max_backward_rotation_rad_mean=float(np.mean(max_back)),
        max_backward_rotation_rad_max=float(np.max(max_back)),
        full_rotation_rate=float(np.mean(rotated)),
        landing_success_rate=float(np.mean(landed)),
        success_rate=float(np.mean(landed)),
        fall_rate=float(np.mean(~upright_end)),
    )
    return out


def is_flip_clip(qpos: np.ndarray, min_rotation_rad: float = np.pi) -> bool:
    """True if the reference root rotates more than ``min_rotation_rad`` about its lateral axis."""
    return bool(np.max(np.abs(cumulative_pitch_rad(qpos[None, :, 3:7]))) > min_rotation_rad)


def summarize_tracking(
    reward: np.ndarray, done: np.ndarray, qpos: np.ndarray, ref_qpos: np.ndarray, ctrl_dt_s: float
) -> dict[str, float]:
    """Reference-tracking metrics over the alive part of each episode.

    ``ref_qpos`` is the reference at the time of every recorded step. Errors are
    averaged over alive steps per episode, then over episodes:
    root position error (m), root orientation error (geodesic, rad), joint RMSE
    (rad), survival time (s) and the fraction of episodes that never terminated.
    """
    mask = alive_mask(done)
    pos_err = np.linalg.norm(qpos[..., 0:3] - ref_qpos[..., 0:3], axis=-1)
    ang_err = quat.quat_angle_between(qpos[..., 3:7], ref_qpos[..., 3:7], np)
    joint_rmse = np.sqrt(np.mean(np.square(qpos[..., 7:] - ref_qpos[..., 7:]), axis=-1))
    out = summarize_common(reward, done, ctrl_dt_s)
    out.update(
        survival_time_s_mean=float(np.mean(episode_lengths(done)) * ctrl_dt_s),
        completion_rate=float(np.mean(~terminated_early(done))),
        root_pos_error_m=float(np.nanmean(_masked_mean(pos_err, mask))),
        root_orientation_error_rad=float(np.nanmean(_masked_mean(ang_err, mask))),
        joint_rmse_rad=float(np.nanmean(_masked_mean(joint_rmse, mask))),
    )
    return out
