"""SuperMimic training loop for the DeepMimic task envs (``T1Mimic``, ``G1Mimic``) (JAX / Flax / Optax).

Follows Algorithm A.1 of Liao et al. (arXiv:2610.28480):

1. **Collect**: the deterministic policy plus Gaussian action noise
   ``N(0, 0.1^2)`` runs in the MJX simulator; transitions go to a
   fixed-size per-environment ring buffer.
2. **Fit the world model** on ``k = 8``-step autoregressive windows seeded at
   logged states and rolled out on logged actions (Eq. 1, state term only).
3. **Improve the policy** by backpropagating the return of ``H = 32``-step
   imagined rollouts through the frozen world model (Eq. 2), bootstrapped by a
   critic fit to GAE(lambda) targets of the imagined rollout (Eq. 3).

Adaptations to this repository (all documented in the README):

* single reference clip instead of a 129k-motion library, so the goal is the
  clip phase (as in the PPO ``mimic`` task) instead of 10 future frames;
* the reward, termination and observation "heads" are **analytic**: they are
  evaluated on the predicted state with differentiable MJX kinematics and the
  exact functions the simulator env uses (``MimicTask.reward_and_termination``).
  The paper allows analytic heads; it learns a soft termination probability,
  here termination is a hard indicator with a stop-gradient (rewards are dense);
* no domain randomisation, so the world model has no sim-context input;
* reward = the DeepMimic reward of ``MimicTask`` (the paper uses BeyondMimic's); with the opt-in routine
  reward (``humanoid_mimic_rl.routine``) imagination adds a differentiable surrogate of the milestone bonuses
  (``env.routine_soft_step``), the simulator env pays the hard bonuses;
* return-based reset sampling is replaced by the env's uniform RSI.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.training import types as training_types
from brax.training.acme import running_statistics
from flax import struct
from mujoco import mjx

from humanoid_mimic_rl import metrics, reference
from humanoid_mimic_rl.envs.backflip import phase_features
from humanoid_mimic_rl.envs.mimic import MimicTask
from humanoid_mimic_rl.supermimic import networks
from humanoid_mimic_rl.supermimic.config import SuperMimicConfig
from humanoid_mimic_rl.supermimic import state as st

RunningStats = running_statistics.RunningStatisticsState


@struct.dataclass
class Replay:
    """Per-environment ring buffer; row ``t`` holds the state *before* step ``t``."""

    qpos: jax.Array  # (L, N, nq)
    qvel: jax.Array  # (L, N, nv)
    time: jax.Array  # (L, N) clip time (s)
    last_act: jax.Array  # (L, N, nu)
    last_last_act: jax.Array  # (L, N, nu)
    action: jax.Array  # (L, N, nu) executed (noisy) action
    done: jax.Array  # (L, N) transition ended the episode
    ptr: jax.Array  # () next row to write
    size: jax.Array  # () rows written, capped at L


def replay_init(capacity: int, num_envs: int, nq: int, nv: int, nu: int) -> Replay:
    z = functools.partial(jp.zeros, dtype=jp.float32)
    return Replay(
        qpos=z((capacity, num_envs, nq)),
        qvel=z((capacity, num_envs, nv)),
        time=z((capacity, num_envs)),
        last_act=z((capacity, num_envs, nu)),
        last_last_act=z((capacity, num_envs, nu)),
        action=z((capacity, num_envs, nu)),
        done=jp.zeros((capacity, num_envs), dtype=bool),
        ptr=jp.zeros((), jp.int32),
        size=jp.zeros((), jp.int32),
    )


def replay_add(buf: Replay, row: dict[str, jax.Array]) -> Replay:
    capacity = buf.qpos.shape[0]
    upd = {k: getattr(buf, k).at[buf.ptr].set(v) for k, v in row.items()}
    return buf.replace(**upd, ptr=(buf.ptr + 1) % capacity, size=jp.minimum(buf.size + 1, capacity))


def replay_windows(buf: Replay, key: jax.Array, batch: int, k: int) -> dict[str, jax.Array]:
    """``batch`` windows of ``k`` transitions: states ``(B, k+1, .)``, actions/done ``(B, k, .)``."""
    capacity, num_envs = buf.qpos.shape[:2]
    k_env, k_off = jax.random.split(key)
    env = jax.random.randint(k_env, (batch,), 0, num_envs)
    off = jax.random.randint(k_off, (batch,), 0, jp.maximum(buf.size - k, 1))
    rows = st.ring_rows(buf.ptr, buf.size, capacity, off[:, None] + jp.arange(k + 1), jp)
    e = env[:, None]
    return {
        "qpos": buf.qpos[rows, e],
        "qvel": buf.qvel[rows, e],
        "time": buf.time[rows, e],
        "action": buf.action[rows[:, :k], e],
        "done": buf.done[rows[:, :k], e],
    }


def replay_states(buf: Replay, key: jax.Array, batch: int) -> dict[str, jax.Array]:
    capacity, num_envs = buf.qpos.shape[:2]
    k_env, k_off = jax.random.split(key)
    env = jax.random.randint(k_env, (batch,), 0, num_envs)
    rows = st.ring_rows(buf.ptr, buf.size, capacity, jax.random.randint(k_off, (batch,), 0, buf.size), jp)
    return {
        "qpos": buf.qpos[rows, env],
        "qvel": buf.qvel[rows, env],
        "time": buf.time[rows, env],
        "last_act": buf.last_act[rows, env],
        "last_last_act": buf.last_last_act[rows, env],
    }


@struct.dataclass
class TrainState:
    wm_params: Any
    wm_opt: Any
    pi_params: Any
    pi_opt: Any
    v_params: Any
    v_opt: Any
    obs_stats: RunningStats
    feat_stats: RunningStats
    out_stats: RunningStats
    critic_stats: RunningStats
    replay: Replay
    env_state: Any
    ep_return: jax.Array  # (N,) running episode return of the collectors
    key: jax.Array


def _norm(x: jax.Array, stats: RunningStats) -> jax.Array:
    return running_statistics.normalize(x, stats)


def _update_stats(stats: RunningStats, x: jax.Array, valid: jax.Array) -> RunningStats:
    return running_statistics.update(stats, x, weights=valid.astype(jp.float32))


class SuperMimicTrainer:
    """Builds and runs the jitted SuperMimic iteration for a ``MimicTask`` env (T1 or G1)."""

    def __init__(self, env: MimicTask, cfg: SuperMimicConfig) -> None:
        cfg.validate()
        self.env = env
        self.cfg = cfg
        self.nu = env.action_size
        self.nq = env.mj_model.nq
        self.nv = env.mj_model.nv
        self.period_s = env.episode_end_s
        self.default_pose = env.default_pose
        nu = self.nu
        # Actor-observation noise: sensors only, none on last action / phase.
        self.obs_noise = jp.concatenate([env.obs_noise_half_widths(), jp.zeros(nu + 3)])
        self.obs_dim = int(self.obs_noise.shape[0])
        self.feat_dim = st.num_features(nu)
        self.out_dim = st.num_outputs(nu)
        self.critic_dim = 1 + 3 + 3 + 3 + nu + nu + nu + 3

        # Differentiable kinematics always uses the pure-JAX MJX backend.
        self._kin_model = mjx.put_model(env.mj_model, impl="jax")
        self._kin_data = mjx.make_data(env.mj_model, impl="jax")

        self.wm = networks.MLP(cfg.wm_hidden, self.out_dim, layer_norm=True)
        self.policy = networks.Policy(cfg.policy_hidden, nu)
        self.critic = networks.Critic(cfg.critic_hidden)
        schedule = optax.cosine_decay_schedule(
            cfg.learning_rate, int(cfg.lr_decay_fraction * cfg.num_iterations), cfg.lr_final_ratio
        )
        self.wm_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip_global_norm), optax.adam(schedule))
        self.pi_tx = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip_global_norm),
            optax.clip(cfg.policy_grad_clip_elementwise),
            optax.adam(schedule, b1=0.9, b2=0.95),
        )
        self.v_tx = optax.chain(optax.clip_by_global_norm(cfg.grad_clip_global_norm), optax.adam(schedule))

        self._v_reset = jax.vmap(env.reset)
        self._v_step = jax.vmap(env.step)
        self._v_targets = jax.vmap(env.motor_targets)
        self._v_reward = jax.vmap(env.reward_and_termination)
        if env.routine_enabled:  # imagination-only differentiable surrogate of the milestone bonuses
            self._v_routine_init = jax.vmap(env.routine_soft_init)
            self._v_routine_step = jax.vmap(env.routine_soft_step)

    # ------------------------------------------------------------------
    # Analytic heads.
    # ------------------------------------------------------------------

    def kinematics(self, qpos: jax.Array) -> tuple[jax.Array, jax.Array]:
        """End-effector positions and trunk/head geom heights for one ``qpos``."""
        d = mjx.kinematics(self._kin_model, self._kin_data.replace(qpos=qpos))
        return d.site_xpos[self.env.ee_site_ids], d.geom_xpos[self.env.fall_geom_ids, 2]

    def actor_obs(self, qpos: jax.Array, qvel: jax.Array, last_act: jax.Array, t: jax.Array) -> jax.Array:
        return st.actor_obs(qpos, qvel, last_act, t, self.default_pose, self.period_s, jp)

    def critic_obs(self, qpos: jax.Array, qvel: jax.Array, last_act: jax.Array, t: jax.Array) -> jax.Array:
        return st.critic_features(qpos, qvel, last_act, t, self.period_s, jp)

    def wm_step(self, wm_params: Any, ts: TrainState, qpos: jax.Array, qvel: jax.Array, u: jax.Array):
        feat = st.features(qpos, qvel, u - qpos[..., 7:], jp)
        out_n = self.wm.apply(wm_params, _norm(feat, ts.feat_stats))
        out = out_n * ts.out_stats.std + ts.out_stats.mean
        return st.integrate(qpos, qvel, out, jp)

    # ------------------------------------------------------------------
    # Initialisation and self-check.
    # ------------------------------------------------------------------

    def init(self, seed: int) -> TrainState:
        key = jax.random.PRNGKey(seed)
        key, k_wm, k_pi, k_v, k_env = jax.random.split(key, 5)
        cfg = self.cfg
        wm_params = self.wm.init(k_wm, jp.zeros(self.feat_dim))
        pi_params = self.policy.init(k_pi, jp.zeros(self.obs_dim))
        v_params = self.critic.init(k_v, jp.zeros(self.critic_dim))
        env_state = jax.jit(self._v_reset)(jax.random.split(k_env, cfg.num_envs))
        return TrainState(
            wm_params=wm_params,
            wm_opt=self.wm_tx.init(wm_params),
            pi_params=pi_params,
            pi_opt=self.pi_tx.init(pi_params),
            v_params=v_params,
            v_opt=self.v_tx.init(v_params),
            obs_stats=running_statistics.init_state(jp.zeros(self.obs_dim)),
            feat_stats=running_statistics.init_state(jp.zeros(self.feat_dim)),
            out_stats=running_statistics.init_state(jp.zeros(self.out_dim)),
            critic_stats=running_statistics.init_state(jp.zeros(self.critic_dim)),
            replay=replay_init(cfg.buffer_steps_per_env, cfg.num_envs, self.nq, self.nv, self.nu),
            env_state=env_state,
            ep_return=jp.zeros(cfg.num_envs),
            key=key,
        )

    def self_check(self, seed: int, tol: float = 1e-3) -> dict[str, float]:
        """Verify the analytic heads reproduce the simulator env's functions.

        Steps one env with a random action, re-runs ``mjx.forward`` on the
        result (MuJoCo computes sensors and site/geom poses *before* the last
        substep's integration, so raw step data lags ``qpos``/``qvel`` by one
        substep) and compares, on that consistent data: (a) the analytic actor
        observation with the env's noise-free observation entries, (b) the
        differentiable kinematics with MJX positions, (c) the shared reward on
        both. Raises ``RuntimeError`` above ``tol``. The returned
        ``*_lag_vs_env`` entries quantify the one-substep sensor lag of the
        real env relative to the analytic heads (informational).
        """
        env, nu = self.env, self.nu
        k0, k1 = jax.random.split(jax.random.PRNGKey(seed))
        s0 = jax.jit(env.reset)(k0)
        action = jax.random.uniform(k1, (nu,), minval=-0.5, maxval=0.5)
        s1 = jax.jit(env.step)(s0, action)
        d = jax.jit(mjx.forward)(env.mjx_model, s1.data)
        obs_fwd = jax.jit(lambda dd, info: env.proprio_obs(dd, info, phase_features(dd.time, self.period_s)))(
            d, dict(s1.info)
        )
        obs = self.actor_obs(d.qpos, d.qvel, action, d.time)
        state_obs, priv = obs_fwd["state"], obs_fwd["privileged_state"]
        n_state = state_obs.shape[-1]
        if n_state != self.obs_dim:
            raise RuntimeError(f"env actor obs has {n_state} dims, analytic obs {self.obs_dim}")
        sensors = 6 + 2 * nu
        ee, fall_h = jax.jit(self.kinematics)(d.qpos)
        r_kin, _, _ = env.reward_and_termination(
            d.qpos, d.qvel, ee, fall_h, d.time, action, s0.info["last_act"], s0.info["last_last_act"]
        )
        r_fwd, _, _ = env.reward_and_termination(
            d.qpos, d.qvel, d.site_xpos[env.ee_site_ids], d.geom_xpos[env.fall_geom_ids, 2], d.time,
            action, s0.info["last_act"], s0.info["last_last_act"],
        )
        checked = {
            "obs_sensors": float(jp.max(jp.abs(obs[:sensors] - priv[n_state : n_state + sensors]))),
            "obs_tail": float(jp.max(jp.abs(obs[sensors:] - state_obs[sensors:]))),
            "kinematics_m": float(
                jp.maximum(
                    jp.max(jp.abs(ee - d.site_xpos[env.ee_site_ids])),
                    jp.max(jp.abs(fall_h - d.geom_xpos[env.fall_geom_ids, 2])),
                )
            ),
            "reward": float(jp.abs(r_kin - r_fwd)),
        }
        if max(checked.values()) > tol:
            raise RuntimeError(f"SuperMimic analytic heads disagree with the env: {checked}")
        raw_priv = s1.obs["privileged_state"]
        lag = {
            "obs_lag_vs_env": float(jp.max(jp.abs(obs[:sensors] - raw_priv[n_state : n_state + sensors]))),
            "reward_lag_vs_env": float(jp.abs(r_kin - s1.reward)),
        }
        return {**checked, **lag}

    # ------------------------------------------------------------------
    # 1. Collection.
    # ------------------------------------------------------------------

    def _collect_step(self, ts: TrainState, _: None) -> tuple[TrainState, dict[str, jax.Array]]:
        cfg = self.cfg
        key, k_noise, k_reset = jax.random.split(ts.key, 3)
        es = ts.env_state
        obs = es.obs["state"]
        pre = es.data
        last_act, last_last_act = es.info["last_act"], es.info["last_last_act"]
        a_det = self.policy.apply(ts.pi_params, _norm(obs, ts.obs_stats))
        action = jp.clip(a_det + cfg.action_noise_std * jax.random.normal(k_noise, a_det.shape), -1.0, 1.0)
        u = self._v_targets(pre.time, action)
        feat = st.features(pre.qpos, pre.qvel, u - pre.qpos[:, 7:], jp)
        crit = self.critic_obs(pre.qpos, pre.qvel, last_act, pre.time)

        nxt = self._v_step(es, action)
        done = nxt.done > 0.5
        post = nxt.data
        target = st.targets_from_transition(pre.qpos, pre.qvel, post.qpos, post.qvel, jp)
        finite = jp.all(jp.isfinite(target), axis=-1) & jp.all(jp.isfinite(feat), axis=-1)

        replay = replay_add(
            ts.replay,
            {
                "qpos": pre.qpos,
                "qvel": pre.qvel,
                "time": pre.time,
                "last_act": last_act,
                "last_last_act": last_last_act,
                "action": action,
                "done": done | ~finite,
            },
        )
        fresh = self._v_reset(jax.random.split(k_reset, cfg.num_envs))

        def pick(a: jax.Array, b: jax.Array) -> jax.Array:
            mask = jp.reshape(done, done.shape + (1,) * (a.ndim - 1))
            return jp.where(mask, a, b)

        env_state = jax.tree.map(pick, fresh, nxt)
        ep_return = ts.ep_return + nxt.reward
        finished_return = jp.where(done, ep_return, jp.nan)
        ts = ts.replace(
            replay=replay,
            env_state=env_state,
            ep_return=jp.where(done, 0.0, ep_return),
            key=key,
            obs_stats=_update_stats(ts.obs_stats, obs, jp.ones(cfg.num_envs, bool)),
            feat_stats=_update_stats(ts.feat_stats, jp.nan_to_num(feat), finite),
            out_stats=_update_stats(ts.out_stats, jp.nan_to_num(target), finite),
            critic_stats=_update_stats(ts.critic_stats, crit, jp.ones(cfg.num_envs, bool)),
        )
        return ts, {"finished_return": finished_return, "done": done.astype(jp.float32)}

    def collect(self, ts: TrainState) -> tuple[TrainState, dict[str, jax.Array]]:
        ts, out = jax.lax.scan(self._collect_step, ts, None, length=self.cfg.env_steps_per_iteration)
        n_done = jp.sum(out["done"])
        mean_ret = jp.nansum(out["finished_return"]) / jp.maximum(n_done, 1.0)
        return ts, {"collect/episodes": n_done, "collect/episode_return": mean_ret}

    # ------------------------------------------------------------------
    # 2. World model.
    # ------------------------------------------------------------------

    def _wm_loss(self, wm_params: Any, ts: TrainState, w: dict[str, jax.Array]) -> jax.Array:
        k = self.cfg.wm_window
        valid = st.window_valid_mask(w["done"], jp)
        qpos, qvel = w["qpos"][:, 0], w["qvel"][:, 0]
        total = 0.0
        for j in range(k):
            u = self._v_targets(w["time"][:, j], w["action"][:, j])
            qpos, qvel = self.wm_step(wm_params, ts, qpos, qvel, u)
            err = st.state_error(qpos, qvel, w["qpos"][:, j + 1], w["qvel"][:, j + 1], jp) / ts.out_stats.std
            sq = jp.mean(jp.square(err), axis=-1)
            m = valid[:, j] & jp.isfinite(sq)
            total = total + jp.sum(jp.where(m, sq, 0.0)) / jp.maximum(jp.sum(m), 1)
        return total / k

    def _wm_batch(self, ts: TrainState, _: None) -> tuple[TrainState, jax.Array]:
        key, k = jax.random.split(ts.key)
        w = replay_windows(ts.replay, k, self.cfg.wm_batch_size, self.cfg.wm_window)
        loss, grads = jax.value_and_grad(self._wm_loss)(ts.wm_params, ts, w)
        updates, wm_opt = self.wm_tx.update(grads, ts.wm_opt, ts.wm_params)
        return ts.replace(wm_params=optax.apply_updates(ts.wm_params, updates), wm_opt=wm_opt, key=key), loss

    def fit_world_model(self, ts: TrainState) -> tuple[TrainState, dict[str, jax.Array]]:
        ts, losses = jax.lax.scan(self._wm_batch, ts, None, length=self.cfg.wm_batches_per_iteration)
        return ts, {"wm/loss": jp.mean(losses)}

    # ------------------------------------------------------------------
    # 3. Policy and critic through imagined rollouts.
    # ------------------------------------------------------------------

    def _imagine(self, pi_params: Any, ts: TrainState, start: dict[str, jax.Array], key: jax.Array):
        cfg, dt = self.cfg, self.env.dt
        v_kin = jax.vmap(self.kinematics)

        def body(carry, k_step):
            qpos, qvel, t, la, lla, alive, rstate = carry
            obs = self.actor_obs(qpos, qvel, la, t)
            obs = obs + self.obs_noise * jax.random.uniform(k_step, obs.shape, minval=-1.0, maxval=1.0)
            action = self.policy.apply(pi_params, _norm(obs, ts.obs_stats))
            u = self._v_targets(t, action)
            qpos2, qvel2 = self.wm_step(ts.wm_params, ts, qpos, qvel, u)
            t2 = t + dt
            ee, fall_h = v_kin(qpos2)
            reward, _, flags = self._v_reward(qpos2, qvel2, ee, fall_h, t2, action, la, lla)
            if self.env.routine_enabled:
                rstate, extra = self._v_routine_step(rstate, qpos2, ee, t2, flags["finished"], flags["fell"])
                reward = reward + extra
            done = jax.lax.stop_gradient(flags["done"].astype(jp.float32))
            crit = self.critic_obs(qpos, qvel, la, t)
            out = {"reward": reward, "done": done, "alive": alive, "critic_obs": crit}
            return (qpos2, qvel2, t2, action, la, alive * (1.0 - done), rstate), out

        if cfg.remat_imagination:
            body = jax.checkpoint(body)
        b = start["qpos"].shape[0]
        rstate0 = self._v_routine_init(start["qpos"], start["time"]) if self.env.routine_enabled else {}
        carry0 = (
            start["qpos"], start["qvel"], start["time"], start["last_act"], start["last_last_act"], jp.ones(b), rstate0
        )
        carry, traj = jax.lax.scan(body, carry0, jax.random.split(key, cfg.horizon))
        qpos_h, qvel_h, t_h, la_h, _, alive_h, _ = carry
        crit_h = self.critic_obs(qpos_h, qvel_h, la_h, t_h)
        return traj, crit_h, alive_h  # traj leaves are (H, B, ...)

    def _policy_loss(self, pi_params: Any, ts: TrainState, start: dict[str, jax.Array], key: jax.Array):
        cfg = self.cfg
        traj, crit_h, alive_h = self._imagine(pi_params, ts, start, key)
        disc = cfg.discount ** jp.arange(cfg.horizon)
        v_h = self.critic.apply(ts.v_params, _norm(crit_h, ts.critic_stats))
        ret = jp.sum(disc[:, None] * traj["alive"] * traj["reward"], axis=0) + (
            cfg.discount**cfg.horizon * alive_h * v_h
        )
        return -jp.mean(ret), (traj, crit_h, ret)

    def _critic_targets(self, ts: TrainState, traj: dict[str, jax.Array], crit_h: jax.Array):
        """GAE(lambda) targets (Eq. 3), computed backwards from ``R_H = V_H``."""
        cfg = self.cfg
        values = self.critic.apply(ts.v_params, _norm(traj["critic_obs"], ts.critic_stats))  # (H, B)
        v_h = self.critic.apply(ts.v_params, _norm(crit_h, ts.critic_stats))
        next_values = jp.concatenate([values[1:], v_h[None]], axis=0)

        def back(r_next, x):
            reward, done, v_next = x
            r = reward + cfg.discount * (1.0 - done) * ((1.0 - cfg.gae_lambda) * v_next + cfg.gae_lambda * r_next)
            return r, r

        _, targets = jax.lax.scan(back, v_h, (traj["reward"], traj["done"], next_values), reverse=True)
        return jax.lax.stop_gradient(targets)

    def _critic_fit(self, ts: TrainState, inputs: jax.Array, targets: jax.Array, weights: jax.Array, key: jax.Array):
        cfg = self.cfg
        n = inputs.shape[0]
        mb = n // cfg.critic_minibatches

        def loss_fn(v_params, x, y, w):
            v = self.critic.apply(v_params, _norm(x, ts.critic_stats))
            return jp.sum(w * jp.square(v - y)) / jp.maximum(jp.sum(w), 1.0)

        def minibatch(carry, idx):
            v_params, v_opt = carry
            loss, g = jax.value_and_grad(loss_fn)(v_params, inputs[idx], targets[idx], weights[idx])
            upd, v_opt = self.v_tx.update(g, v_opt, v_params)
            return (optax.apply_updates(v_params, upd), v_opt), loss

        def epoch(carry, k):
            perm = jax.random.permutation(k, n)[: mb * cfg.critic_minibatches].reshape(cfg.critic_minibatches, mb)
            return jax.lax.scan(minibatch, carry, perm)

        (v_params, v_opt), losses = jax.lax.scan(
            epoch, (ts.v_params, ts.v_opt), jax.random.split(key, cfg.critic_epochs)
        )
        return ts.replace(v_params=v_params, v_opt=v_opt), jp.mean(losses)

    def improve_policy(self, ts: TrainState, update_policy: bool = True) -> tuple[TrainState, dict[str, jax.Array]]:
        """Policy step on imagined rollouts, then the critic fit on them.

        ``update_policy=False`` (static) still fits the critic but leaves the policy and its optimiser
        state unchanged: used after ``warm_start`` so a fresh world model and critic do not move a trained
        policy before they are fit to it.
        """
        cfg = self.cfg
        key, k_start, k_img, k_crit = jax.random.split(ts.key, 4)
        start = replay_states(ts.replay, k_start, cfg.policy_batch_size)
        (loss, (traj, crit_h, ret)), grads = jax.value_and_grad(self._policy_loss, has_aux=True)(
            ts.pi_params, ts, start, k_img
        )
        if update_policy:
            updates, pi_opt = self.pi_tx.update(grads, ts.pi_opt, ts.pi_params)
            ts = ts.replace(pi_params=optax.apply_updates(ts.pi_params, updates), pi_opt=pi_opt, key=key)
        else:
            ts = ts.replace(key=key)

        traj = jax.lax.stop_gradient(traj)
        targets = self._critic_targets(ts, traj, jax.lax.stop_gradient(crit_h))
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731  (H, B, ...) -> (H*B, ...)
        obs_f, y, w = flat(traj["critic_obs"]), flat(targets), flat(traj["alive"])
        ts, v_loss = self._critic_fit(ts, obs_f, y, w, k_crit)
        # Rewards are scaled by the control dt, so returns (and the MSE) are small;
        # explained variance is the scale-free measure of critic quality.
        v = self.critic.apply(ts.v_params, _norm(obs_f, ts.critic_stats))
        w_sum = jp.maximum(jp.sum(w), 1.0)
        y_mean = jp.sum(w * y) / w_sum
        var_y = jp.sum(w * jp.square(y - y_mean)) / w_sum
        resid = y - v
        var_r = jp.sum(w * jp.square(resid - jp.sum(w * resid) / w_sum)) / w_sum
        return ts, {
            "policy/imagined_return": jp.mean(ret),
            "policy/grad_norm": optax.global_norm(grads),
            "policy/imagined_alive_frac": jp.mean(traj["alive"][-1]),
            "critic/loss": v_loss,
            "critic/target_std": jp.sqrt(var_y),
            "critic/explained_variance": 1.0 - var_r / jp.maximum(var_y, 1e-12),
        }

    # ------------------------------------------------------------------
    # Jitted entry points.
    # ------------------------------------------------------------------

    def jitted(self) -> tuple[Callable[[TrainState], Any], Callable[[TrainState], Any]]:
        """``(warmup_fn, iteration_fn)``: collection only, and collect + fit WM + improve policy."""

        def warmup(ts: TrainState):
            return self.collect(ts)

        def iteration(ts: TrainState):
            ts, m_c = self.collect(ts)
            ts, m_w = self.fit_world_model(ts)
            ts, m_p = self.improve_policy(ts)
            return ts, {**m_c, **m_w, **m_p}

        return jax.jit(warmup), jax.jit(iteration)

    def jitted_frozen_policy(self) -> Callable[[TrainState], Any]:
        """Collect + fit WM + fit critic, policy unchanged (the phase after ``warm_start``)."""

        def iteration(ts: TrainState):
            ts, m_c = self.collect(ts)
            ts, m_w = self.fit_world_model(ts)
            ts, m_p = self.improve_policy(ts, update_policy=False)
            return ts, {**m_c, **m_w, **m_p}

        return jax.jit(iteration)

    def warm_start(self, ts: TrainState, checkpoint: Path, stats_count: float = 1e7) -> TrainState:
        """Load policy weights and observation statistics from a ``save_checkpoint`` file.

        The checkpoint holds only the policy and the observation mean/std, so the world model and critic
        start fresh (fit them with ``jitted_frozen_policy`` first). The statistics get a large pseudo-count
        so the policy's input normalisation stays put while collection keeps updating it. Raises if the
        network shapes do not match (different env, clip type or trick set).
        """
        payload = flax.serialization.msgpack_restore(checkpoint.read_bytes())
        missing = {"policy", "obs_mean", "obs_std"} - set(payload)
        if missing:
            raise ValueError(f"{checkpoint}: not a SuperMimic checkpoint (missing {sorted(missing)})")
        pi_params = flax.serialization.from_state_dict(ts.pi_params, payload["policy"])
        for new, old in zip(jax.tree_util.tree_leaves(pi_params), jax.tree_util.tree_leaves(ts.pi_params)):
            if new.shape != old.shape:
                raise ValueError(f"{checkpoint}: policy shape {new.shape} != {old.shape} of this config")
        mean, std = jp.asarray(payload["obs_mean"]), jp.asarray(payload["obs_std"])
        if mean.shape != ts.obs_stats.mean.shape:
            raise ValueError(f"{checkpoint}: obs dim {mean.shape} != {ts.obs_stats.mean.shape} of this env")
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
            raise ValueError(f"{checkpoint}: non-finite observation statistics")
        if not 0 < stats_count < 2**32:
            raise ValueError(f"stats_count must be in (0, 2**32), got {stats_count}")
        old = ts.obs_stats.count
        if isinstance(old, training_types.UInt64):  # brax keeps the count as two uint32 halves
            count = training_types.UInt64(hi=jp.zeros_like(jp.asarray(old.hi)),
                                          lo=jp.asarray(int(stats_count), jp.asarray(old.lo).dtype))
        else:
            count = jp.asarray(stats_count, jp.asarray(old).dtype)
        obs_stats = ts.obs_stats.replace(mean=mean, std=std, count=count,
                                         summed_variance=jp.square(std) * jp.float32(stats_count))
        return ts.replace(pi_params=pi_params, obs_stats=obs_stats)

    # ------------------------------------------------------------------
    # Evaluation and checkpoints.
    # ------------------------------------------------------------------

    def make_eval_fn(self, eval_env: MimicTask) -> Callable[[Any, RunningStats, jax.Array], dict[str, jax.Array]]:
        """Deterministic rollouts of ``num_eval_envs`` episodes in the simulator (no auto-reset)."""
        n, steps = self.cfg.num_eval_envs, int(eval_env.unwrapped._config.episode_length)  # pylint: disable=protected-access
        v_reset, v_step = jax.vmap(eval_env.reset), jax.vmap(eval_env.step)

        @jax.jit
        def run(pi_params, obs_stats, key):
            s = v_reset(jax.random.split(key, n))

            def body(s, _):
                a = self.policy.apply(pi_params, _norm(s.obs["state"], obs_stats))
                s = v_step(s, a)
                rec = {
                    "reward": s.reward,
                    "done": s.done,
                    "qpos": s.data.qpos,
                    "fell": s.metrics["fell"],
                    "tracking_failed": s.metrics["tracking_failed"],
                    **{k: s.metrics[k] for k in metrics.ROUTINE_FLAGS if k in s.metrics},
                    "time": s.data.time,
                    "up_z": jax.vmap(eval_env.up_z)(s.data),
                    "feet_contact": jax.vmap(eval_env.feet_contact)(s.data),
                }
                return s, rec

            _, traj = jax.lax.scan(body, s, None, length=steps)
            return traj

        return run

    def save_checkpoint(self, path: Path, ts: TrainState, iteration: int) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy": flax.serialization.to_state_dict(jax.device_get(ts.pi_params)),
            "obs_mean": np.asarray(ts.obs_stats.mean),
            "obs_std": np.asarray(ts.obs_stats.std),
        }
        out = path / f"{iteration:08d}.msgpack"
        out.write_bytes(flax.serialization.msgpack_serialize(payload))
        return out


def summarize_eval(
    traj: dict[str, jax.Array], ctrl_dt_s: float, home_height_m: float, clip: reference.ReferenceClip
) -> dict[str, float]:
    """Eval metrics from a ``(T, N, ...)`` rollout: tracking metrics for every clip
    (``metrics.summarize_tracking``), plus flip metrics when the clip is a flip and routine rates when the
    env records the routine milestone flags."""
    tr = {k: np.moveaxis(np.asarray(v), 0, 1) for k, v in traj.items()}
    ref = reference.sample_qpos_batch(clip, tr["time"])
    out = metrics.summarize_tracking(
        tr["reward"], tr["done"], tr["qpos"], ref, ctrl_dt_s, tr["fell"], tr["tracking_failed"]
    )
    if metrics.is_flip_clip(clip.qpos):
        flip = metrics.summarize_flip(
            tr["reward"], tr["done"], tr["qpos"][..., 3:7], tr["up_z"], tr["qpos"][..., 2], tr["feet_contact"],
            ctrl_dt_s, 0.5, 0.8, 0.75 * home_height_m,
        )
        # The flip "fall_rate" (not upright at the end, or terminated) is a different quantity.
        flip["flip_fall_rate"] = flip.pop("fall_rate")
        out.update(flip)
    if all(k in tr for k in metrics.ROUTINE_FLAGS):
        out.update(metrics.summarize_routine(tr["done"], *(tr[k] for k in metrics.ROUTINE_FLAGS)))
    return out


def load_policy(checkpoint: Path, cfg_json: Path) -> Callable[[dict[str, jax.Array], jax.Array], tuple[jax.Array, dict]]:
    """Inference function ``policy(obs, key) -> (action, {})`` for ``eval_render.py``."""
    cfg = json.loads(cfg_json.read_text(encoding="utf-8"))["supermimic"]
    payload = flax.serialization.msgpack_restore(checkpoint.read_bytes())
    obs_dim = int(np.asarray(payload["obs_mean"]).shape[0])
    nu = (obs_dim - 9) // 3  # obs = gyro 3 + gravity 3 + 3 * nu + phase 3
    policy = networks.Policy(tuple(cfg["policy_hidden"]), nu)
    params = payload["policy"]
    mean, std = jp.asarray(payload["obs_mean"]), jp.asarray(payload["obs_std"])

    def act(obs: dict[str, jax.Array], key: jax.Array) -> tuple[jax.Array, dict]:
        del key  # deterministic policy
        return policy.apply(params, (obs["state"] - mean) / std), {}

    return act
