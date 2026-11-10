"""Multi-stage "routine" milestones and per-milestone bonuses (backflip -> spin kick -> stand).

Everything here is backend-agnostic (``xp`` = numpy or jax.numpy) so the exact logic that runs in the
simulator env (``envs.mimic.MimicTask``) and in SuperMimic's imagination is unit-tested on numpy arrays.

Milestones are computed from the **simulated** robot state, with time windows (seconds of clip time) taken
from the routine clip's ``.report.json`` (``report["routine"]``, written by ``scripts/make_g1_routine.py``):

* ``flip_done``: cumulative backward pitch of the pelvis (rotation about the body y axis, signed, counted
  only for steps whose clip time is inside the flip window) reaches ``flip_min_rotation_rad`` AND the robot has
  landed -- both feet in contact, pelvis ``up_z > land_min_up_z``, not fallen -- at some step within
  ``flip_landing_grace_s`` after the end of the flip window (and after the rotation was reached);
* ``kick_done``: cumulative yaw (rotation about world z, signed in the reference spin direction, counted inside
  the kick window) reaches ``kick_min_rotation_rad`` AND the kicking foot site reached at least
  ``kick_min_height_frac`` x the reference's peak height of that site inside the kick window;
* ``standing``: pelvis ``up_z > stand_min_up_z`` and not fallen; ``routine_done = flip_done & kick_done &
  standing`` (the score: ``metrics.summarize_routine``).

Reward (``envs.mimic.MimicTask.step``; ``reward_and_termination`` is NOT changed). The dense DeepMimic
tracking reward is kept and each milestone gets its own bonus, ``<x>_bonus_fraction`` x (return of a
perfect-tracking episode, ``sum(positive reward scales) * ctrl_dt * steps``):

* flip bonus: on the step ``flip_done`` is first completed (the landing);
* kick bonus: on the first step after the kick window closes, if ``kick_done``;
* stand bonus: on the final step of a normal finish (not fell, not tracking-failed, not invalid) with
  ``standing``;
* optional ``gated`` (off by default): the dense reward of every step after a milestone window is multiplied
  by whether that milestone was hit (flip: after ``flip_end + grace``; kick: after ``kick_end``).

The env metrics ``flip_done`` / ``kick_done`` / ``routine_done`` are edge indicators: 1 on the single step on
which the milestone is first completed / on which a normal finish has ``routine_done``, else 0. Brax sums
metrics over an episode, so the PPO ``eval/episode_flip_done`` etc. are directly per-episode rates.

Reference-state initialisation can start an episode inside or after a milestone window. With
``credit_skipped_milestones`` (default) the milestone state is initialised from a numpy replay of the
reference through the same update, so progress the reference state already contains is credited (otherwise
only the ~10 % of episodes that start at t = 0 could ever earn a bonus). Evaluation starts at t = 0
(``rsi_start_prob = 1``), so the reported rates are not inflated by this.

SuperMimic imagination: ``SuperMimicTrainer`` evaluates ``MimicTask.reward_and_termination`` (stateless, dense
only) inside its imagined rollouts, and its replay buffer stores no rewards, so the hard bonuses above never
reach SuperMimic's policy or critic; they only appear in the simulator-side ``collect/episode_return`` and
evaluation ``episode_return``. To give SuperMimic a gradient, ``soft_update`` / ``soft_reward`` provide a
differentiable surrogate used by the trainer in imagination only: potentials
``phi_flip = max_t sigmoid(rot - thr) * landed_soft`` and ``phi_kick = max_t sigmoid(yaw - thr) *
sigmoid(foot_max - thr)`` (running max latches), rewarded as ``bonus * (phi_t - phi_{t-1})`` (potential
differences, so the episode sum telescopes to ``bonus * phi_final``), plus ``stand_bonus * sigmoid(up_z - thr)``
at the final step. ``landed_soft`` uses the foot site heights below ``soft_foot_contact_z_m`` instead of the
contact sensors (not available in imagination). Imagination starts from replay states whose true milestone
progress is unknown, so the soft state is credited from the *reference* at the start time (same
``credit_at``) and imagined rotation increments are added on top. Every sigmoid is shifted by
``SOFT_MARGIN`` scale units so a quantity exactly at its threshold scores 0.88, and inputs are clipped so
there is no overflow / NaN path.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import numpy as np

from humanoid_mimic_rl import quat

Array = Any
SOFT_MARGIN = 2.0  # sigmoid shift in scale units: value at the threshold is sigmoid(2) = 0.88
SOFT_CLIP = 30.0  # sigmoid argument clip (no exp overflow)


@dataclasses.dataclass(frozen=True)
class RoutineThresholds:
    """Milestone thresholds and bonus weights (units in the field names); see the module docstring."""

    flip_min_rotation_rad: float = math.radians(300.0)
    flip_landing_grace_s: float = 0.5
    land_min_up_z: float = 0.8  # pelvis z axis . world z; 0.8 = 37 deg tilt
    kick_min_rotation_rad: float = math.radians(270.0)
    kick_min_height_frac: float = 0.6  # of the reference kick-foot peak height (m, world z)
    stand_min_up_z: float = 0.8
    flip_bonus_fraction: float = 0.1  # each bonus = fraction x perfect-tracking episode return
    kick_bonus_fraction: float = 0.1
    stand_bonus_fraction: float = 0.1
    gated: bool = False
    credit_skipped_milestones: bool = True
    # SuperMimic imagination surrogate only (see module docstring)
    soft_rotation_scale_rad: float = math.radians(30.0)
    soft_height_scale_m: float = 0.05
    soft_up_z_scale: float = 0.1
    soft_foot_contact_z_m: float = 0.07  # foot site height below which a foot counts as on the ground

    def validate(self) -> None:
        for name in ("flip_min_rotation_rad", "kick_min_rotation_rad"):
            v = getattr(self, name)
            if not (math.isfinite(v) and 0.0 < v <= 4.0 * math.pi):
                raise ValueError(f"routine.{name} must be in (0, 4 pi] rad, got {v}")
        if not (math.isfinite(self.flip_landing_grace_s) and 0.0 <= self.flip_landing_grace_s <= 5.0):
            raise ValueError(f"routine.flip_landing_grace_s must be in [0, 5] s, got {self.flip_landing_grace_s}")
        for name in ("land_min_up_z", "stand_min_up_z"):
            v = getattr(self, name)
            if not (math.isfinite(v) and 0.0 < v < 1.0):
                raise ValueError(f"routine.{name} must be in (0, 1), got {v}")
        if not 0.0 < self.kick_min_height_frac <= 1.0:
            raise ValueError(f"routine.kick_min_height_frac must be in (0, 1], got {self.kick_min_height_frac}")
        for name in ("flip_bonus_fraction", "kick_bonus_fraction", "stand_bonus_fraction"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError(f"routine.{name} must be >= 0, got {v}")
        for name in ("soft_rotation_scale_rad", "soft_height_scale_m", "soft_up_z_scale", "soft_foot_contact_z_m"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"routine.{name} must be > 0, got {v}")


def thresholds_from_config(cfg: Mapping[str, Any]) -> RoutineThresholds:
    """Build and validate thresholds from the env's ``routine`` config section."""
    fields = {f.name for f in dataclasses.fields(RoutineThresholds)}
    thr = RoutineThresholds(**{k: cfg[k] for k in fields})
    thr.validate()
    return thr


@dataclasses.dataclass(frozen=True)
class RoutineWindows:
    """Clip-time windows (s) and kick description of a routine clip (``report["routine"]``)."""

    flip_start_s: float
    flip_end_s: float
    kick_start_s: float
    kick_end_s: float
    kick_foot_site: int  # index into the clip's site names
    kick_foot_peak_height_m: float  # reference peak world z of that site inside the kick window
    kick_yaw_sign: float  # +1 / -1: direction of the reference spin about world z

    def validate(self) -> None:
        vals = (self.flip_start_s, self.flip_end_s, self.kick_start_s, self.kick_end_s, self.kick_foot_peak_height_m)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"routine windows must be finite: {self}")
        if not (0.0 <= self.flip_start_s < self.flip_end_s and 0.0 <= self.kick_start_s < self.kick_end_s):
            raise ValueError(f"routine windows must have 0 <= start < end: {self}")
        if not (self.flip_start_s < self.kick_start_s and self.flip_end_s < self.kick_end_s):
            raise ValueError(f"the flip window must come before the kick window: {self}")
        if not self.kick_foot_peak_height_m > 0.0:
            raise ValueError(f"kick foot peak height must be > 0 m, got {self.kick_foot_peak_height_m}")
        if self.kick_yaw_sign not in (-1.0, 1.0):
            raise ValueError(f"kick_yaw_sign must be +/-1, got {self.kick_yaw_sign}")


def windows_from_report(report: Mapping[str, Any], site_names: tuple[str, ...]) -> RoutineWindows:
    """Windows from a clip report; raises ``ValueError`` if it has no (valid) ``routine`` section."""
    sec = report.get("routine")
    if not isinstance(sec, Mapping):
        raise ValueError("clip report has no 'routine' section; build the clip with scripts/make_g1_routine.py")
    try:
        foot = str(sec["kick_foot_site"])
        win = RoutineWindows(
            flip_start_s=float(sec["flip_window_s"][0]),
            flip_end_s=float(sec["flip_window_s"][1]),
            kick_start_s=float(sec["kick_window_s"][0]),
            kick_end_s=float(sec["kick_window_s"][1]),
            kick_foot_site=site_names.index(foot),
            kick_foot_peak_height_m=float(sec["kick_foot_peak_height_m"]),
            kick_yaw_sign=float(sec["kick_yaw_sign"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid 'routine' section in clip report: {exc!r}") from exc
    win.validate()
    return win


# ----------------------------------------------------------------------
# Rotation accumulation and the milestone update (xp-generic).
# ----------------------------------------------------------------------


def rotation_increments(q_prev: Array, q_cur: Array, xp: ModuleType) -> tuple[Array, Array]:
    """``(d_pitch_body_rad, d_yaw_world_rad)`` between two root quaternions ``(..., 4)`` (wxyz, world <- body).

    ``d_pitch_body`` is the body-frame y component of ``log(q_prev^-1 q_cur)`` (negative = backward, as
    ``metrics.cumulative_pitch_rad``); ``d_yaw_world`` the world-z component of ``log(q_cur q_prev^-1)``.
    Both are continuous across the +/-pi wrap of Euler angles (they are per-step relative rotations) and
    are 0 for a non-finite input. Valid while one step rotates less than pi.
    """
    body = quat.quat_to_rotvec(quat.quat_mul(quat.quat_conj(q_prev, xp), q_cur, xp), xp)[..., 1]
    world = quat.quat_to_rotvec(quat.quat_mul(q_cur, quat.quat_conj(q_prev, xp), xp), xp)[..., 2]
    return xp.where(xp.isfinite(body), body, 0.0), xp.where(xp.isfinite(world), world, 0.0)


def initial_state(root_quat: Array, xp: ModuleType, credit: Mapping[str, Array] | None = None) -> dict[str, Array]:
    """Fresh milestone state at episode start; ``credit`` overrides entries (RSI credit, see module doc)."""
    zero = xp.zeros(root_quat.shape[:-1], dtype=xp.float32)
    false = zero > 1.0
    state = {
        "prev_quat": root_quat,
        "flip_rot_rad": zero,
        "flip_rot_hit": false,
        "flip_landed": false,
        "kick_yaw_rad": zero,
        "kick_yaw_hit": false,
        "kick_foot_hit": false,
    }
    if credit:
        state.update({k: xp.asarray(v, dtype=state[k].dtype) for k, v in credit.items()})
    return state


def update(
    state: Mapping[str, Array],
    t_s: Array,
    root_quat: Array,
    up_z: Array,
    feet_contact: Array,
    kick_foot_z_m: Array,
    fell: Array,
    thr: RoutineThresholds,
    win: RoutineWindows,
    xp: ModuleType,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """One step: returns ``(new_state, flags)`` with booleans ``flip_done``, ``kick_done``, ``standing``, ``routine_done``.

    ``t_s`` is the clip time of the state reached by the step, ``feet_contact`` the (left, right) foot-floor
    contact booleans ``(..., 2)``, ``kick_foot_z_m`` the world height of the kicking foot site and ``fell`` the
    env's fall flag for this step.
    """
    fell = xp.asarray(fell, dtype=bool)
    feet_contact = xp.asarray(feet_contact, dtype=bool)
    d_pitch, d_yaw = rotation_increments(state["prev_quat"], root_quat, xp)
    in_flip = (t_s > win.flip_start_s) & (t_s <= win.flip_end_s)
    flip_rot = state["flip_rot_rad"] + xp.where(in_flip, -d_pitch, 0.0)
    flip_hit = state["flip_rot_hit"] | (in_flip & (flip_rot >= thr.flip_min_rotation_rad))
    in_landing = (t_s > win.flip_start_s) & (t_s <= win.flip_end_s + thr.flip_landing_grace_s)
    landed_now = xp.all(feet_contact, axis=-1) & (up_z > thr.land_min_up_z) & ~fell
    flip_landed = state["flip_landed"] | (flip_hit & in_landing & landed_now)

    in_kick = (t_s > win.kick_start_s) & (t_s <= win.kick_end_s)
    kick_yaw = state["kick_yaw_rad"] + xp.where(in_kick, win.kick_yaw_sign * d_yaw, 0.0)
    yaw_hit = state["kick_yaw_hit"] | (in_kick & (kick_yaw >= thr.kick_min_rotation_rad))
    foot_hit = state["kick_foot_hit"] | (in_kick & (kick_foot_z_m >= thr.kick_min_height_frac * win.kick_foot_peak_height_m))

    flip_done = flip_hit & flip_landed
    kick_done = yaw_hit & foot_hit
    standing = (up_z > thr.stand_min_up_z) & ~fell
    new_state = {
        "prev_quat": root_quat,
        "flip_rot_rad": flip_rot.astype(state["flip_rot_rad"].dtype),
        "flip_rot_hit": flip_hit,
        "flip_landed": flip_landed,
        "kick_yaw_rad": kick_yaw.astype(state["kick_yaw_rad"].dtype),
        "kick_yaw_hit": yaw_hit,
        "kick_foot_hit": foot_hit,
    }
    flags = {
        "flip_done": flip_done,
        "kick_done": kick_done,
        "standing": standing,
        "routine_done": flip_done & kick_done & standing,
    }
    return new_state, flags


def normal_finish(finished: Array, fell: Array, tracking_failed: Array, invalid: Array) -> Array:
    """True on the final step of an episode that was not cut short by a fall, tracking failure or invalid state."""
    return finished & ~fell & ~tracking_failed & ~invalid


def bonus_reward(
    old_state: Mapping[str, Array],
    flags: Mapping[str, Array],
    t_s: Array,
    ctrl_dt_s: float,
    finish_ok: Array,
    bonus_flip: float,
    bonus_kick: float,
    bonus_stand: float,
    win: RoutineWindows,
    xp: ModuleType,
) -> Array:
    """Sum of the milestone bonuses earned at this step (0 on every other step; finite by construction).

    ``flags`` are the ``update`` flags of this step, ``old_state`` the milestone state before it,
    ``finish_ok`` = ``normal_finish``. Flip: the step ``flip_done`` is first true. Kick: the first step after the
    kick window closed (``kick_end + dt/2 < t <= kick_end + 3 dt/2``) with ``kick_done``; an episode that starts after the
    window never pays it. Stand: normal finish and ``standing``.
    """
    flip_new = flags["flip_done"] & ~(old_state["flip_rot_hit"] & old_state["flip_landed"])
    closes_s = win.kick_end_s + 0.5 * ctrl_dt_s  # half a step of slack: robust to float32 clip time on the step grid
    kick_closed_now = (t_s > closes_s) & (t_s - ctrl_dt_s <= closes_s)
    paid_kick = flags["kick_done"] & kick_closed_now
    paid_stand = finish_ok & flags["standing"]
    return (
        xp.where(flip_new, bonus_flip, 0.0) + xp.where(paid_kick, bonus_kick, 0.0) + xp.where(paid_stand, bonus_stand, 0.0)
    )


def gate(t_s: Array, flip_done: Array, kick_done: Array, thr: RoutineThresholds, win: RoutineWindows, xp: ModuleType) -> Array:
    """1 while no milestone window has closed unmet, else 0 (``gated`` dense-reward multiplier)."""
    flip_ok = (t_s <= win.flip_end_s + thr.flip_landing_grace_s) | flip_done
    kick_ok = (t_s <= win.kick_end_s) | kick_done
    return xp.where(flip_ok & kick_ok, 1.0, 0.0)


# ----------------------------------------------------------------------
# Reference replay (numpy): rotation progress of the clip itself, for RSI credit and clip reports.
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReferenceProgress:
    """Per-frame milestone progress of the reference clip (all ``(T,)``)."""

    flip_rot_rad: np.ndarray
    kick_yaw_rad: np.ndarray
    kick_foot_hit: np.ndarray  # reference kick foot has reached the height threshold inside the window
    kick_foot_max_z_m: np.ndarray  # running max of the kick foot site height inside the kick window (0 before)


def reference_progress(
    qpos: np.ndarray, kick_foot_z_m: np.ndarray, fps_hz: float, thr: RoutineThresholds, win: RoutineWindows
) -> ReferenceProgress:
    """Replay the rotation / height accumulation on the reference ``qpos`` (``(T, 7 + n)``)."""
    n = qpos.shape[0]
    t = np.arange(n) / fps_hz
    d_pitch, d_yaw = rotation_increments(qpos[:-1, 3:7], qpos[1:, 3:7], np)
    t1 = t[1:]
    in_flip = (t1 > win.flip_start_s) & (t1 <= win.flip_end_s)
    in_kick = (t1 > win.kick_start_s) & (t1 <= win.kick_end_s)
    flip = np.concatenate([[0.0], np.cumsum(np.where(in_flip, -d_pitch, 0.0))])
    yaw = np.concatenate([[0.0], np.cumsum(np.where(in_kick, win.kick_yaw_sign * d_yaw, 0.0))])
    in_kick_all = (t > win.kick_start_s) & (t <= win.kick_end_s)
    high = in_kick_all & (kick_foot_z_m >= thr.kick_min_height_frac * win.kick_foot_peak_height_m)
    running_max = np.maximum.accumulate(np.where(in_kick_all, kick_foot_z_m, 0.0))
    return ReferenceProgress(flip, yaw, np.cumsum(high) > 0, running_max)


def credit_at(
    progress: ReferenceProgress, k: Array, t_s: Array, thr: RoutineThresholds, win: RoutineWindows, xp: ModuleType
) -> dict[str, Array]:
    """Milestone state credited to an episode starting on reference frame ``k`` (clip time ``t_s``)."""
    flip_rot = xp.asarray(progress.flip_rot_rad)[k]
    yaw = xp.asarray(progress.kick_yaw_rad)[k]
    return {
        "flip_rot_rad": flip_rot,
        "flip_rot_hit": flip_rot >= thr.flip_min_rotation_rad,
        "flip_landed": (flip_rot >= thr.flip_min_rotation_rad) & (t_s >= win.flip_end_s),
        "kick_yaw_rad": yaw,
        "kick_yaw_hit": yaw >= thr.kick_min_rotation_rad,
        "kick_foot_hit": xp.asarray(progress.kick_foot_hit)[k],
    }


# ----------------------------------------------------------------------
# Differentiable surrogate for SuperMimic imagination (see module docstring).
# ----------------------------------------------------------------------


def _soft_ge(x: Array, threshold: Array, scale: float, xp: ModuleType) -> Array:
    """Smooth ``x >= threshold`` in (0, 1): sigmoid((x - threshold) / scale + SOFT_MARGIN), input clipped."""
    z = xp.clip((x - threshold) / scale + SOFT_MARGIN, -SOFT_CLIP, SOFT_CLIP)
    return 1.0 / (1.0 + xp.exp(-z))


def soft_initial_state(
    root_quat: Array, t_s: Array, progress: ReferenceProgress, k: Array, thr: RoutineThresholds, win: RoutineWindows,
    xp: ModuleType,
) -> dict[str, Array]:
    """Soft state for an imagined rollout starting at clip time ``t_s`` (reference frame ``k``): accumulators and
    latches credited from the reference at that time (the robot's own history is unknown there)."""
    credit = credit_at(progress, k, t_s, thr, win, xp)
    f32 = xp.float32
    foot_max = xp.asarray(progress.kick_foot_max_z_m)[k].astype(f32)
    yaw = credit["kick_yaw_rad"].astype(f32)
    flip_phi = xp.where(credit["flip_landed"], 1.0, 0.0).astype(f32)
    kick_phi = (
        _soft_ge(yaw, thr.kick_min_rotation_rad, thr.soft_rotation_scale_rad, xp)
        * _soft_ge(foot_max, thr.kick_min_height_frac * win.kick_foot_peak_height_m, thr.soft_height_scale_m, xp)
    ).astype(f32)
    return {
        "prev_quat": root_quat,
        "flip_rot_rad": credit["flip_rot_rad"].astype(f32),
        "kick_yaw_rad": yaw,
        "kick_foot_max_z_m": foot_max,
        "flip_phi": flip_phi,
        "kick_phi": kick_phi,
    }


def soft_update(
    state: Mapping[str, Array],
    t_s: Array,
    root_quat: Array,
    up_z: Array,
    feet_z_m: Array,
    kick_foot_z_m: Array,
    thr: RoutineThresholds,
    win: RoutineWindows,
    xp: ModuleType,
) -> tuple[dict[str, Array], Array, Array]:
    """One imagined step: ``(new_state, d_phi_flip, d_phi_kick)``, the potential increments (>= 0).

    ``feet_z_m`` are the (left, right) foot site heights ``(..., 2)``; the landing factor is the product of the
    soft "both feet within ``soft_foot_contact_z_m`` of the floor" and soft ``up_z > land_min_up_z``, active only
    inside the landing window. Gradients flow through the quaternion, up_z and site heights.
    """
    d_pitch, d_yaw = rotation_increments(state["prev_quat"], root_quat, xp)
    in_flip = (t_s > win.flip_start_s) & (t_s <= win.flip_end_s)
    flip_rot = state["flip_rot_rad"] + xp.where(in_flip, -d_pitch, 0.0)
    in_landing = (t_s > win.flip_start_s) & (t_s <= win.flip_end_s + thr.flip_landing_grace_s)
    low_feet = -feet_z_m  # soft "z <= limit" as soft ">=" on -z
    on_ground = xp.prod(_soft_ge(low_feet, -thr.soft_foot_contact_z_m, thr.soft_height_scale_m, xp), axis=-1)
    landed = xp.where(in_landing, on_ground * _soft_ge(up_z, thr.land_min_up_z, thr.soft_up_z_scale, xp), 0.0)
    flip_inst = _soft_ge(flip_rot, thr.flip_min_rotation_rad, thr.soft_rotation_scale_rad, xp) * landed
    flip_phi = xp.maximum(state["flip_phi"], flip_inst)

    in_kick = (t_s > win.kick_start_s) & (t_s <= win.kick_end_s)
    kick_yaw = state["kick_yaw_rad"] + xp.where(in_kick, win.kick_yaw_sign * d_yaw, 0.0)
    foot_max = xp.where(in_kick, xp.maximum(state["kick_foot_max_z_m"], kick_foot_z_m), state["kick_foot_max_z_m"])
    kick_inst = _soft_ge(kick_yaw, thr.kick_min_rotation_rad, thr.soft_rotation_scale_rad, xp) * _soft_ge(
        foot_max, thr.kick_min_height_frac * win.kick_foot_peak_height_m, thr.soft_height_scale_m, xp
    )
    kick_phi = xp.maximum(state["kick_phi"], kick_inst)
    new_state = {
        "prev_quat": root_quat,
        "flip_rot_rad": flip_rot.astype(state["flip_rot_rad"].dtype),
        "kick_yaw_rad": kick_yaw.astype(state["kick_yaw_rad"].dtype),
        "kick_foot_max_z_m": foot_max.astype(state["kick_foot_max_z_m"].dtype),
        "flip_phi": flip_phi.astype(state["flip_phi"].dtype),
        "kick_phi": kick_phi.astype(state["kick_phi"].dtype),
    }
    return new_state, flip_phi - state["flip_phi"], kick_phi - state["kick_phi"]


def soft_stand(up_z: Array, finished: Array, fell: Array, thr: RoutineThresholds, xp: ModuleType) -> Array:
    """Soft ``standing`` in [0, 1] on a final, not-fallen step; 0 elsewhere."""
    ok = xp.asarray(finished, dtype=bool) & ~xp.asarray(fell, dtype=bool)
    return xp.where(ok, _soft_ge(up_z, thr.stand_min_up_z, thr.soft_up_z_scale, xp), 0.0)


def perfect_return(positive_scales_sum: float, ctrl_dt_s: float, episode_end_s: float) -> float:
    """Return of a perfect-tracking episode: every term at its maximum of 1 for every control step."""
    return positive_scales_sum * ctrl_dt_s * math.ceil(episode_end_s / ctrl_dt_s)
