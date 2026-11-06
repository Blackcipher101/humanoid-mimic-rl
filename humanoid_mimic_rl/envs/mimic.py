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

import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from humanoid_mimic_rl import reference
from humanoid_mimic_rl import rewards
from humanoid_mimic_rl import t1_model
from humanoid_mimic_rl.envs import t1_full_collision as base
from humanoid_mimic_rl.envs.backflip import phase_features


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
        impl="jax",
        naconmax=32 * 8192,
        njmax=160,
    )


def episode_length_for(clip: reference.ReferenceClip, ctrl_dt_s: float, hold_final_s: float) -> int:
    """Control steps needed to play the whole clip plus the final hold."""
    return int(math.ceil((clip.duration_s + hold_final_s) / ctrl_dt_s))


class T1Mimic(base.T1FullCollisionEnv):
    """Track ``config.reference_path`` with DeepMimic reward, RSI and ET."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config or default_config(), config_overrides)
        if not self._config.reference_path:
            raise ValueError("T1Mimic requires config.reference_path")
        clip = reference.load_clip(Path(self._config.reference_path))
        if clip.joint_names != t1_model.T1_JOINT_NAMES:
            raise ValueError("reference joint order does not match the T1 model")
        if clip.site_names != t1_model.T1_END_EFFECTOR_SITES:
            raise ValueError(f"reference sites {clip.site_names} != {t1_model.T1_END_EFFECTOR_SITES}")
        self._clip = clip
        self._ref_qpos = jp.array(clip.qpos)
        self._ref_qvel = jp.array(clip.qvel)
        self._ref_sites = jp.array(clip.site_pos)
        self._ee_site_ids = np.array([self._mj_model.site(s).id for s in clip.site_names])
        self._episode_end_s = clip.duration_s + self._config.hold_final_s

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
        obs = self.proprio_obs(data, info, phase_features(data.time, self._episode_end_s))
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

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
        done = flags["done"] | self.is_invalid(data)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        obs = self.proprio_obs(data, state.info, phase_features(t, self._episode_end_s))
        for k, v in scaled.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["root_pos_err_m"] = flags["root_pos_err_m"]
        state.metrics["fell"] = flags["fell"].astype(jp.float32)
        state.metrics["tracking_failed"] = flags["tracking_failed"].astype(jp.float32)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))
