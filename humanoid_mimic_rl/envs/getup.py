"""T1 get-up: recover from lying on the ground (any orientation) to standing.

Design follows playground's ``Go1Getup`` adapted to a humanoid:

* reset start states (v3; v1/v2 never produced a standing start, see below):
  - with ``fallen_init_prob``: dropped from ``drop_height_m`` with a uniformly random orientation and random joint
    angles, settled for ``settle_time_s`` while holding those angles;
  - with ``push_init_prob``: standing in the home pose with a random push, snapshotted at a random point of the
    ensuing fall (0.02 / 0.25 / 0.5 / 1.0 s), so crouching / kneeling / half-up states appear (reverse curriculum);
  - otherwise: standing still in the home pose (snapshot after 0.02 s), so the policy also learns to stay up.
  v1/v2 gave every start a +-0.5 m/s, +-0.5 rad/s root kick and a 1 s settle, so even the "standing" starts were
  lying down before the first action (measured: torso 0.03 m, up_z 0.06) and the standing gate never opened;
* actions are joint-position offsets added to the **current** joint angles
  (Go1Getup found this works better than offsets from the home pose for
  recovery because it gives a wide initial range of motion);
* reward = upright + torso height + (gated) home posture + (gated) stillness
  - regularisers. The gate opens when the torso is upright and high.

Episodes never terminate early (except on NaN), so the policy must get up
within ``episode_length`` control steps.

Brax's auto-reset restores the cached initial ``data``/``obs`` but not
``info``; nothing in ``info`` affects rewards except the action history,
matching playground's tasks.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

from humanoid_mimic_rl import rewards
from humanoid_mimic_rl import t1_model
from humanoid_mimic_rl.envs import t1_full_collision as base


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,  # s (50 Hz policy)
        sim_dt=0.002,  # s, T1 MJX model default
        episode_length=400,  # control steps = 8 s
        action_repeat=1,
        action_scale=0.5,  # rad per unit action
        soft_joint_pos_limit_factor=0.95,
        fallen_init_prob=0.55,
        push_init_prob=0.3,  # rest: standing still
        push_lin_mps=0.8,  # max |root linear velocity| of the push
        push_ang_radps=1.5,  # max |root angular velocity| of the push
        drop_height_m=0.6,
        settle_time_s=1.0,
        init_clearance_m=0.02,
        stand_min_up_z=0.9,  # ~25 deg max torso tilt counts as upright
        stand_height_frac=0.9,  # of the home torso height
        max_joint_vel_radps=10.0,
        posture_gain=0.5,
        stand_still_gain=0.5,
        feet_under_length_m=0.25,  # length scale of the feet-under-torso reward
        noise_config=base.noise_config(),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # v2 (after the first full run sat upright on the floor, 0/128 stood up): uprightness only pays
                # with height, height is squared, and feet-under-torso shapes the sit -> squat -> stand path.
                upright=1.0,
                torso_height=2.0,
                feet_under=1.0,
                posture=1.0,
                stand_still=0.5,
                action_rate=-0.0005,  # v4: -0.01 cost ~-223 per episode vs ~+95 for all positive terms (v3 eval: 0/128 rose), so the policy learned not to move
                torques=-1e-5,
                dof_pos_limits=-1.0,
                dof_vel=-0.01,
            ),
        ),
        impl="jax",
        naconmax=32 * 8192,  # total contacts across worlds (warp impl only)
        njmax=160,  # constraint rows per world (warp impl only)
    )


class T1Getup(base.T1FullCollisionEnv):
    """Get up from the ground. Observations: ``state`` (actor), ``privileged_state`` (critic)."""

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config or default_config(), config_overrides)
        settle = int(round(self._config.settle_time_s / self.sim_dt))
        cuts = [int(round(t / self.sim_dt)) for t in (0.02, 0.25, 0.5)] + [settle]
        # step counts between snapshot times 0.02 / 0.25 / 0.5 / settle_time_s
        self._snapshot_steps = [b - a for a, b in zip([0] + cuts[:-1], cuts)]
        self._stand_height_m = self._config.stand_height_frac * t1_model.T1_HOME_ROOT_HEIGHT_M
        self._foot_site_ids = np.array([self._mj_model.site(name).id for name in ("left_foot", "right_foot")])

    def _random_fallen_qpos(self, rng: jax.Array) -> jax.Array:
        rng_q, rng_j = jax.random.split(rng)
        quat = jax.random.normal(rng_q, (4,))
        quat = quat / (jp.linalg.norm(quat) + 1e-9)
        joints = jax.random.uniform(rng_j, (self.action_size,), minval=self._soft_lowers, maxval=self._soft_uppers)
        qpos = self._home_qpos.at[2].set(self._config.drop_height_m)
        return qpos.at[3:7].set(quat).at[7:].set(joints)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, k_mode, k_pose, k_vel, k_snap = jax.random.split(rng, 5)
        u = jax.random.uniform(k_mode)
        fallen = u < self._config.fallen_init_prob
        pushed = (~fallen) & (u < self._config.fallen_init_prob + self._config.push_init_prob)
        qpos = jp.where(fallen, self._random_fallen_qpos(k_pose), self._home_qpos)
        kick = jax.random.uniform(k_vel, (6,), minval=-1.0, maxval=1.0)
        push = kick * jp.array([self._config.push_lin_mps] * 3 + [self._config.push_ang_radps] * 3)
        root_vel = jp.where(fallen, 0.5 * kick, jp.where(pushed, push, jp.zeros(6)))
        qvel = jp.zeros(self.mjx_model.nv).at[0:6].set(root_vel)
        data = self.make_data(qpos, qvel, qpos[7:])
        data = jax.tree.map(lambda a, b: jp.where(fallen, a, b),
                            self.lift_above_floor(data, self._config.init_clearance_m), data)
        # Simulate the settle / fall in chunks and keep one snapshot: same cost as one full settle.
        ctrl = qpos[7:]
        snaps = []
        for n in self._snapshot_steps:
            data = mjx_env.step(self.mjx_model, data, ctrl, n)
            snaps.append(data)
        snap_idx = jp.where(fallen, len(snaps) - 1,
                            jp.where(pushed, jax.random.randint(k_snap, (), 0, len(snaps)), 0))
        data = jax.tree.map(lambda *xs: jp.stack(xs)[snap_idx], *snaps)
        data = data.replace(time=jp.zeros_like(data.time))

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales}
        metrics["standing"] = jp.zeros(())
        obs = self.proprio_obs(data, info, jp.zeros(0))
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        motor_targets = state.data.qpos[7:] + action * self._config.action_scale
        data = self.physics_step(state.data, motor_targets)

        up_z = self.up_z(data)
        height = self.torso_height(data)
        gate = rewards.standing_gate(up_z, height, self._config.stand_min_up_z, self._stand_height_m, jp)
        weights = jp.ones(self.action_size)
        terms = {
            "upright": rewards.tall_upright_reward(up_z, height, t1_model.T1_HOME_ROOT_HEIGHT_M, jp),
            "torso_height": jp.square(rewards.height_reward(height, t1_model.T1_HOME_ROOT_HEIGHT_M, jp)),
            # gated by uprightness: ungated, curling up while lying on the side scored (getup v2 eval video)
            "feet_under": jp.any(self.feet_contact(data)).astype(height.dtype) * jp.clip(up_z, 0.0, 1.0)
            * rewards.feet_under_torso_reward(
                data.site_xpos[self._imu_site_id, :2],
                data.site_xpos[self._foot_site_ids, :2],
                self._config.feet_under_length_m,
                jp,
            ),
            "posture": gate
            * rewards.posture_reward(data.qpos[7:], self._default_pose, weights, self._config.posture_gain, jp),
            "stand_still": gate
            * rewards.exp_kernel(jp.sum(jp.square(action)), self._config.stand_still_gain, jp),
            "action_rate": rewards.action_rate_cost(action, state.info["last_act"], state.info["last_last_act"], jp),
            "torques": rewards.torque_cost(data.actuator_force, jp),
            "dof_pos_limits": rewards.joint_limit_cost(data.qpos[7:], self._soft_lowers, self._soft_uppers, jp),
            "dof_vel": rewards.joint_velocity_excess_cost(data.qvel[6:], self._config.max_joint_vel_radps, jp),
        }
        reward, scaled = base.weighted_reward(terms, self._config.reward_config.scales, self.dt)
        done = self.is_invalid(data)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        obs = self.proprio_obs(data, state.info, jp.zeros(0))
        for k, v in scaled.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["standing"] = gate
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype))
