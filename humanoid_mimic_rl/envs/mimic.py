"""DeepMimic-style reference-motion tracking for the T1 (Peng et al., 2018).

Ingredients, as in DeepMimic:

* **imitation reward** ``w_p r_pose + w_v r_vel + w_e r_ee + w_c r_root``
  (``rewards.deepmimic_terms``; root position replaces the CoM term);
* **reference state initialisation (RSI)**: episodes start at a uniformly
  sampled clip time with the reference pose and velocity;
* **early termination (ET)**: the episode ends when the trunk or head comes
  close to the floor, or when the root drifts too far from the reference.

Deviations (documented): the policy outputs *residual* joint targets around
the reference pose (``q_ref(t + dt) + action * action_scale``) rather than
absolute PD targets, and after the last frame the reference holds the final
pose for ``hold_final_s`` so landing stabilisation is learned.

The clip time is ``data.time``: it is set to the RSI time at reset and is
restored with the cached initial state by Brax's auto-reset, so no per-episode
state lives in ``info`` apart from the action history.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from humanoid_mimic_rl import reference
from humanoid_mimic_rl import rewards
from humanoid_mimic_rl import routine
from humanoid_mimic_rl import t1_model
from humanoid_mimic_rl.envs import t1_full_collision as base
from humanoid_mimic_rl.envs.backflip import phase_features


TERMINATIONS = ("root_pose", "height_tilt")


@dataclasses.dataclass(frozen=True)
class _RoutineRuntime:
    """Resolved routine reward: thresholds, clip windows, reference progress, terminal bonus value."""

    thr: routine.RoutineThresholds
    win: routine.RoutineWindows
    progress: routine.ReferenceProgress
    bonuses: tuple[float, float, float]  # flip, kick, stand bonus (reward units)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        reference_path="",  # .npz written by scripts/make_reference.py; required
        ctrl_dt=0.02,  # s
        sim_dt=0.002,  # s
        episode_length=0,  # control steps; set from the clip by ``episode_length_for``
        action_repeat=1,
        action_scale=0.3,  # rad per unit action, residual on q_ref
        soft_joint_pos_limit_factor=0.95,
        hold_final_s=0.6,
        rsi_start_prob=0.1,  # probability of starting at t = 0 instead of a random time
        rsi_end_margin_s=0.1,  # never start closer than this to the clip end
        init_clearance_m=0.0,
        fall_height_m=0.2,
        max_root_pos_err_m=0.6,
        max_root_angle_err_rad=1.2,
        termination="root_pose",  # "root_pose": 3D root drift / full orientation; "height_tilt": SuperMimic App. A
        max_root_height_err_m=0.25,  # "height_tilt" only
        max_tilt_err=0.8,  # "height_tilt" only: |g_z_body - g_z_ref|, in (0, 2]
        gains=config_dict.create(pose=2.0, velocity=0.1, end_effector=40.0, root=10.0),
        noise_config=base.noise_config(),
        reward_config=config_dict.create(
            scales=config_dict.create(
                pose=0.65,
                velocity=0.1,
                end_effector=0.15,
                root=0.1,
                action_rate=-0.002,
            ),
        ),
        # Opt-in multi-stage routine reward (``humanoid_mimic_rl.routine``); disabled = task unchanged.
        routine=config_dict.create(
            enabled=False,
            report_path="",  # clip report with a "routine" section; required when enabled
            flip_min_rotation_rad=math.radians(300.0),  # cumulative backward pelvis pitch inside the flip window
            flip_landing_grace_s=0.5,  # s after the flip window within which the landing must happen
            land_min_up_z=0.8,  # pelvis z . world z at landing
            kick_min_rotation_rad=math.radians(270.0),  # cumulative yaw inside the kick window
            kick_min_height_frac=0.6,  # kick foot height as a fraction of the reference peak (m)
            stand_min_up_z=0.8,  # pelvis z . world z at the final step
            flip_bonus_fraction=0.1,  # each bonus = fraction x return of a perfect-tracking episode
            kick_bonus_fraction=0.1,
            stand_bonus_fraction=0.1,
            gated=False,  # also multiply the dense reward after a missed milestone window by 0
            credit_skipped_milestones=True,  # RSI starts after a milestone are credited with it
            soft_rotation_scale_rad=math.radians(30.0),  # SuperMimic imagination surrogate only
            soft_height_scale_m=0.05,
            soft_up_z_scale=0.1,
            soft_foot_contact_z_m=0.07,
        ),
        impl="jax",
        naconmax=32 * 8192,
        njmax=160,
    )


def episode_length_for(clip: reference.ReferenceClip, ctrl_dt_s: float, hold_final_s: float) -> int:
    """Control steps needed to play the whole clip plus the final hold."""
    return int(math.ceil((clip.duration_s + hold_final_s) / ctrl_dt_s))


class MimicTask:
    """Robot-agnostic DeepMimic task; combine with a full-collision robot env.

    The concrete class must inherit from a robot base env (``T1FullCollisionEnv``
    or ``G1FullCollisionEnv``) and define ``JOINT_NAMES`` / ``END_EFFECTOR_SITES``.
    """

    JOINT_NAMES: tuple[str, ...]
    END_EFFECTOR_SITES: tuple[str, ...]

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config or default_config(), config_overrides)  # type: ignore[call-arg]
        if not self._config.reference_path:
            raise ValueError(f"{type(self).__name__} requires config.reference_path")
        clip = reference.load_clip(Path(self._config.reference_path))
        if clip.joint_names != self.JOINT_NAMES:
            raise ValueError(f"reference joint order does not match the {type(self).__name__} robot")
        if clip.site_names != self.END_EFFECTOR_SITES:
            raise ValueError(f"reference sites {clip.site_names} != {self.END_EFFECTOR_SITES}")
        c = self._config
        if c.termination not in TERMINATIONS:
            raise ValueError(f"termination must be one of {TERMINATIONS}, got {c.termination!r}")
        if not c.max_root_height_err_m > 0.0:
            raise ValueError(f"max_root_height_err_m must be > 0, got {c.max_root_height_err_m}")
        if not 0.0 < c.max_tilt_err <= 2.0:
            raise ValueError(f"max_tilt_err must be in (0, 2], got {c.max_tilt_err}")
        self._clip = clip
        self._ref_qpos = jp.array(clip.qpos)
        self._ref_qvel = jp.array(clip.qvel)
        self._ref_sites = jp.array(clip.site_pos)
        self._ee_site_ids = np.array([self._mj_model.site(s).id for s in clip.site_names])
        self._episode_end_s = clip.duration_s + self._config.hold_final_s
        self._routine = self._build_routine(clip)

    def _build_routine(self, clip: reference.ReferenceClip) -> _RoutineRuntime | None:
        rc = self._config.routine
        if not rc.enabled:
            return None
        report_path = Path(rc.report_path)
        if not report_path.is_file():
            raise ValueError(f"routine reward needs the clip report, not found: {report_path!r}")
        thr = routine.thresholds_from_config(rc)
        win = routine.windows_from_report(json.loads(report_path.read_text(encoding="utf-8")), clip.site_names)
        if win.kick_end_s > self._episode_end_s:
            raise ValueError(f"kick window ends at {win.kick_end_s} s, after the episode end {self._episode_end_s} s")
        progress = routine.reference_progress(clip.qpos, clip.site_pos[:, win.kick_foot_site, 2], clip.fps_hz, thr, win)
        scales = self._config.reward_config.scales
        positive = float(sum(v for v in scales.values() if v > 0.0))
        if clip.site_names[:2] != ("left_foot", "right_foot"):
            raise ValueError(f"routine reward expects the first two sites to be the feet, got {clip.site_names}")
        perfect = routine.perfect_return(positive, self.dt, self._episode_end_s)
        bonuses = (
            thr.flip_bonus_fraction * perfect, thr.kick_bonus_fraction * perfect, thr.stand_bonus_fraction * perfect
        )
        return _RoutineRuntime(thr, win, progress, bonuses)

    @property
    def clip(self) -> reference.ReferenceClip:
        return self._clip

    def reference_at(self, t_s: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return reference.sample_clip(self._ref_qpos, self._ref_qvel, self._ref_sites, self._clip.fps_hz, t_s, jp)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_start, k_t = jax.random.split(rng, 3)
        t_max = max(self._clip.duration_s - self._config.rsi_end_margin_s, 0.0)
        t0 = jp.where(
            jax.random.bernoulli(k_start, self._config.rsi_start_prob),
            0.0,
            jax.random.uniform(k_t, (), minval=0.0, maxval=t_max),
        )
        qpos, qvel, _ = self.reference_at(t0)
        data = self.make_data(qpos, qvel, qpos[7:])
        data = self.lift_above_floor(data, self._config.init_clearance_m)
        data = data.replace(time=t0.astype(data.time.dtype))
        info = {
            "rng": rng,
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales}
        metrics.update(root_pos_err_m=jp.zeros(()), fell=jp.zeros(()), tracking_failed=jp.zeros(()))
        if self._routine is not None:
            info["routine"] = self._routine_reset(qpos, t0)
            metrics.update(flip_done=jp.zeros(()), kick_done=jp.zeros(()), routine_done=jp.zeros(()))
        obs = self.proprio_obs(data, info, phase_features(data.time, self._episode_end_s))
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def _routine_reset(self, qpos: jax.Array, t0_s: jax.Array) -> dict[str, jax.Array]:
        r = self._routine
        credit = None
        if r.thr.credit_skipped_milestones:
            k = jp.clip(jp.floor(t0_s * self._clip.fps_hz).astype(jp.int32), 0, self._clip.num_frames - 1)
            credit = routine.credit_at(r.progress, k, t0_s, r.thr, r.win, jp)
        return routine.initial_state(qpos[3:7], jp, credit)

    @property
    def routine_bonuses(self) -> tuple[float, float, float]:
        """(flip, kick, stand) bonuses in reward units; zeros when the routine reward is disabled."""
        return (0.0, 0.0, 0.0) if self._routine is None else self._routine.bonuses

    @property
    def routine_enabled(self) -> bool:
        return self._routine is not None

    def routine_soft_init(self, qpos: jax.Array, t_s: jax.Array) -> dict[str, jax.Array]:
        """Soft milestone state for an imagined rollout starting at ``(qpos, t_s)`` (SuperMimic)."""
        r = self._routine
        k = jp.clip(jp.floor(t_s * self._clip.fps_hz).astype(jp.int32), 0, self._clip.num_frames - 1)
        return routine.soft_initial_state(qpos[3:7], t_s, r.progress, k, r.thr, r.win, jp)

    def routine_soft_step(
        self, rstate: dict[str, jax.Array], qpos: jax.Array, ee_pos_m: jax.Array, t_s: jax.Array,
        finished: jax.Array, fell: jax.Array,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Imagination-only differentiable surrogate of the milestone bonuses: ``(new state, extra reward)``.

        Pure function of the imagined state (``ee_pos_m`` = world positions of ``clip.site_names``); see
        ``humanoid_mimic_rl.routine``. Never used by ``step``.
        """
        r = self._routine
        up_z = 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)
        new_state, d_flip, d_kick = routine.soft_update(
            rstate, t_s, qpos[3:7], up_z, ee_pos_m[:2, 2], ee_pos_m[r.win.kick_foot_site, 2], r.thr, r.win, jp
        )
        bonus_flip, bonus_kick, bonus_stand = r.bonuses
        reward = (
            bonus_flip * d_flip + bonus_kick * d_kick
            + bonus_stand * routine.soft_stand(up_z, finished, fell, r.thr, jp)
        )
        return new_state, jp.where(jp.isfinite(reward), reward, 0.0)

    @property
    def episode_end_s(self) -> float:
        return self._episode_end_s

    @property
    def ee_site_ids(self) -> np.ndarray:
        return self._ee_site_ids

    @property
    def fall_geom_ids(self) -> np.ndarray:
        return self._fall_geom_ids

    def motor_targets(self, t_s: jax.Array, action: jax.Array) -> jax.Array:
        """Joint position targets (rad) for the control step starting at ``t_s``."""
        q_next, _, _ = self.reference_at(t_s + self.dt)
        return q_next[7:] + action * self._config.action_scale

    def reward_and_termination(
        self,
        qpos: jax.Array,
        qvel: jax.Array,
        ee_pos_m: jax.Array,
        fall_geom_heights_m: jax.Array,
        t_s: jax.Array,
        action: jax.Array,
        last_act: jax.Array,
        last_last_act: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array], dict[str, jax.Array]]:
        """Reward and termination of the state reached at clip time ``t_s``.

        Pure function of the robot state, so it is shared by ``step`` and by
        the SuperMimic world-model rollouts (``humanoid_mimic_rl.supermimic``).
        ``ee_pos_m`` are the world positions of ``clip.site_names`` and
        ``fall_geom_heights_m`` the centre heights of the trunk/head geoms.
        Returns ``(reward, scaled_terms, flags)`` with boolean flags
        ``fell``, ``tracking_failed``, ``finished`` and their union ``done``.
        """
        c = self._config
        ref_q, ref_v, ref_sites = self.reference_at(t_s)
        root_pos, root_quat = qpos[0:3], qpos[3:7]
        terms = rewards.deepmimic_terms(
            joint_pos_rad=qpos[7:],
            ref_joint_pos_rad=ref_q[7:],
            joint_vel_radps=qvel[6:],
            ref_joint_vel_radps=ref_v[6:],
            root_quat=root_quat,
            ref_root_quat=ref_q[3:7],
            root_pos_m=root_pos,
            ref_root_pos_m=ref_q[0:3],
            ee_pos_m=ee_pos_m,
            ref_ee_pos_m=ref_sites,
            gains={k: c.gains[k] for k in ("pose", "velocity", "end_effector", "root")},
            xp=jp,
        )
        terms["action_rate"] = rewards.action_rate_cost(action, last_act, last_last_act, jp)
        reward, scaled = base.weighted_reward(terms, c.reward_config.scales, self.dt)
        fell = jp.min(fall_geom_heights_m) < c.fall_height_m
        if c.termination == "height_tilt":
            failed = rewards.height_tilt_failed(
                root_pos, ref_q[0:3], root_quat, ref_q[3:7], c.max_root_height_err_m, c.max_tilt_err, jp
            )
        else:
            failed = rewards.tracking_failed(
                root_pos, ref_q[0:3], root_quat, ref_q[3:7], c.max_root_pos_err_m, c.max_root_angle_err_rad, jp
            )
        finished = t_s >= self._episode_end_s
        flags = {
            "fell": fell,
            "tracking_failed": failed,
            "finished": finished,
            "done": fell | failed | finished,
            "root_pos_err_m": jp.linalg.norm(root_pos - ref_q[0:3]),
        }
        return reward, scaled, flags

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = self.physics_step(state.data, self.motor_targets(state.data.time, action))
        t = data.time
        reward, scaled, flags = self.reward_and_termination(
            data.qpos,
            data.qvel,
            data.site_xpos[self._ee_site_ids],
            data.geom_xpos[self._fall_geom_ids, 2],
            t,
            action,
            state.info["last_act"],
            state.info["last_last_act"],
        )
        invalid = self.is_invalid(data)
        done = flags["done"] | invalid
        if self._routine is not None:
            reward, state = self._routine_step(state, data, t, reward, flags, invalid)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        obs = self.proprio_obs(data, state.info, phase_features(t, self._episode_end_s))
        for k, v in scaled.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["root_pos_err_m"] = flags["root_pos_err_m"]
        state.metrics["fell"] = flags["fell"].astype(jp.float32)
        state.metrics["tracking_failed"] = flags["tracking_failed"].astype(jp.float32)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))

    def _routine_step(
        self, state: mjx_env.State, data: mjx.Data, t_s: jax.Array, reward: jax.Array, flags: dict[str, jax.Array],
        invalid: jax.Array,
    ) -> tuple[jax.Array, mjx_env.State]:
        """Update the milestone state in ``info`` and apply the routine reward shaping to ``reward``."""
        r = self._routine
        old = state.info["routine"]
        new_state, rflags = routine.update(
            old, t_s, data.qpos[3:7], self.up_z(data), self.feet_contact(data),
            data.site_xpos[self._ee_site_ids[r.win.kick_foot_site], 2], flags["fell"], r.thr, r.win, jp,
        )
        state.info["routine"] = new_state
        if r.thr.gated:
            reward = reward * routine.gate(t_s, rflags["flip_done"], rflags["kick_done"], r.thr, r.win, jp)
        finish_ok = routine.normal_finish(flags["finished"], flags["fell"], flags["tracking_failed"], invalid)
        bonus = routine.bonus_reward(old, rflags, t_s, self.dt, finish_ok, *r.bonuses, r.win, jp)
        reward = reward + bonus.astype(reward.dtype)
        edge = {  # 1 on the step a milestone is first completed / a normal finish has routine_done
            "flip_done": rflags["flip_done"] & ~(old["flip_rot_hit"] & old["flip_landed"]),
            "kick_done": rflags["kick_done"] & ~(old["kick_yaw_hit"] & old["kick_foot_hit"]),
            "routine_done": rflags["routine_done"] & finish_ok,
        }
        for k, v in edge.items():
            state.metrics[k] = v.astype(jp.float32)
        return reward, state


class T1Mimic(MimicTask, base.T1FullCollisionEnv):
    """DeepMimic tracking on the Booster T1."""

    HOME_ROOT_HEIGHT_M = t1_model.T1_HOME_ROOT_HEIGHT_M
    JOINT_NAMES = t1_model.T1_JOINT_NAMES
    END_EFFECTOR_SITES = t1_model.T1_END_EFFECTOR_SITES
