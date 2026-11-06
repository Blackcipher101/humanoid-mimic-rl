#!/usr/bin/env python3
"""Train the ``mimic`` task with SuperMimic-style world-model BPTT (see humanoid_mimic_rl/supermimic).

Examples:
  python train_supermimic.py --profile smoke
  python train_supermimic.py --profile full [--iterations 20000] [--reference clip.npz]

Outputs go to ``<logdir>/mimic-supermimic-<profile>-<timestamp>/``:
  run_config.json, progress.jsonl (one line per evaluation), checkpoints/<iter>.msgpack, summary.json.
``eval_render.py --run-dir <dir>`` evaluates and renders the latest checkpoint.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

from humanoid_mimic_rl import tasks  # noqa: E402
from humanoid_mimic_rl.supermimic import config as sm_config  # noqa: E402
from humanoid_mimic_rl.supermimic import trainer as sm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
_PACKAGES = ("jax", "jaxlib", "mujoco", "mujoco-mjx", "playground", "brax", "flax", "optax", "warp-lang")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="t1", choices=("t1", "g1"), help="t1: keyframed clip, g1: optimised clip")
    p.add_argument("--profile", default="full", choices=("smoke", "full"))
    p.add_argument("--impl", default="jax", choices=("jax", "warp"), help="MJX backend of the simulator env")
    p.add_argument("--iterations", type=int, default=None, help="override the profile's iteration count")
    p.add_argument("--num-envs", type=int, default=None, help="override simulator envs (GPU memory lever)")
    p.add_argument("--policy-batch-size", type=int, default=None,
                   help="override imagined rollouts per policy update (GPU memory lever)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--reference", "--clip", dest="reference", type=Path, default=None,
                   help="reference clip (.npz)")
    p.add_argument("--torque-scale", type=float, default=None,
                   help="G1 flip tasks: peak-torque multiplier; must match the clip report (default: read from it)")
    p.add_argument("--allow-infeasible-clip", action="store_true",
                   help="G1: train on a clip that FAILED the torque feasibility check (recorded in run_config; "
                        "the robot keeps its torque limits, so the policy only approximates the clip)")
    p.add_argument("--hold-final-s", type=float, default=None,
                   help="hold the last frame this long after the clip (default 0.6 s; 0 for walking)")
    p.add_argument("--termination", default=None, choices=("root_pose", "height_tilt"),
                   help="mimic tasks: root_pose (default: 3D root drift / full orientation) or height_tilt "
                        "(SuperMimic App. A: root height and tilt error only)")
    p.add_argument("--routine", action=argparse.BooleanOptionalAction, default=None,
                   help="g1_mimic: routine reward: dense tracking + per-milestone bonuses (flip -> kick -> stand, "
                        "humanoid_mimic_rl.routine). Default: on iff the clip report has a 'routine' section")
    p.add_argument("--routine-flip-bonus", type=float, default=None,
                   help="routine reward: flip bonus as a fraction of a perfect-tracking episode return (default 0.1)")
    p.add_argument("--routine-kick-bonus", type=float, default=None, help="kick bonus fraction (default 0.1)")
    p.add_argument("--routine-stand-bonus", type=float, default=None, help="stand-at-end bonus fraction (default 0.1)")
    p.add_argument("--routine-gated", action="store_true",
                   help="routine reward: also multiply the dense reward after a missed milestone window by 0 (off)")
    return p.parse_args()


def _versions() -> dict[str, str]:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def _floats(d: dict) -> dict[str, float]:
    return {k: float(np.asarray(v)) for k, v in d.items()}


def main() -> None:
    args = parse_args()
    cfg = sm_config.smoke_config() if args.profile == "smoke" else sm_config.full_config()
    if args.iterations is not None:
        cfg = dataclasses.replace(cfg, num_iterations=args.iterations, eval_every=min(cfg.eval_every, args.iterations))
    if args.num_envs is not None:
        cfg = dataclasses.replace(cfg, num_envs=args.num_envs)
    if args.policy_batch_size is not None:
        cfg = dataclasses.replace(cfg, policy_batch_size=args.policy_batch_size)
    cfg.validate()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    task = "mimic" if args.robot == "t1" else "g1_mimic"
    run_dir = (args.logdir / f"{task}-supermimic-{args.profile}-{stamp}").resolve()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    env = tasks.make_env(task, args.impl, cfg.num_envs, args.reference, hold_final_s=args.hold_final_s,
                         torque_scale=args.torque_scale, termination=args.termination,
                         allow_infeasible_clip=args.allow_infeasible_clip, routine=args.routine,
                         routine_options=tasks.routine_options_from_args(args))
    eval_env = tasks.make_env(task, args.impl, cfg.num_eval_envs, args.reference, hold_final_s=args.hold_final_s,
                         torque_scale=args.torque_scale, termination=args.termination,
                         allow_infeasible_clip=args.allow_infeasible_clip, routine=args.routine,
                         routine_options=tasks.routine_options_from_args(args),
                         eval_mode=True)
    trainer = sm.SuperMimicTrainer(env, cfg)

    routine_cfg = tasks.routine_run_config(env)
    run_config = {
        "task": task,
        "algo": "supermimic",
        "profile": args.profile,
        "impl": args.impl,
        "seed": args.seed,
        "termination": env.unwrapped._config.termination,  # pylint: disable=protected-access
        "supermimic": dataclasses.asdict(cfg),
        "env_config": env.unwrapped._config.to_dict(),  # pylint: disable=protected-access
        "model_label": getattr(env.unwrapped, "model_label", None),
        "allow_infeasible_clip": args.allow_infeasible_clip,
        "routine": routine_cfg,
        "versions": _versions(),
        "devices": [str(d) for d in jax.devices()],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    print(f"run dir: {run_dir}", flush=True)
    if run_config["model_label"]:
        print(f"model: {run_config['model_label']}", flush=True)
    print(f"devices: {run_config['devices']}  versions: {run_config['versions']}", flush=True)

    check = trainer.self_check(args.seed)
    print(f"[self-check] analytic heads vs simulator, max abs error: {check}", flush=True)

    t0 = time.monotonic()
    ts = trainer.init(args.seed)
    warmup_fn, iteration_fn = trainer.jitted()
    eval_fn = trainer.make_eval_fn(eval_env)
    for _ in range(cfg.warmup_iterations):
        ts, _ = warmup_fn(ts)
    jax.block_until_ready(ts.replay.ptr)
    print(f"[warmup] {cfg.warmup_iterations} collection iterations in {time.monotonic() - t0:.1f}s", flush=True)

    env_steps_per_it = cfg.env_steps_per_iteration * cfg.num_envs
    t_loop = time.monotonic()
    t_first: float | None = None
    last: dict[str, float] = {}
    eval_key = jax.random.PRNGKey(args.seed + 1)
    for it in range(1, cfg.num_iterations + 1):
        ts, m = iteration_fn(ts)
        if it == 1:
            jax.block_until_ready(m)
            t_first = time.monotonic()
            print(f"[compile] first iteration (incl. JIT) {t_first - t_loop:.1f}s", flush=True)
        if it % cfg.eval_every != 0 and it != cfg.num_iterations:
            continue
        train_m = _floats(m)
        traj = eval_fn(ts.pi_params, ts.obs_stats, eval_key)
        ev = sm.summarize_eval(traj, env.dt, env.HOME_ROOT_HEIGHT_M, env.clip)
        now = time.monotonic()
        it_s = (it - 1) / (now - t_first) if t_first and now > t_first and it > 1 else float("nan")
        eta = (cfg.num_iterations - it) / it_s if it_s == it_s and it_s > 0 else float("nan")
        ckpt = trainer.save_checkpoint(ckpt_dir, ts, it)
        last = {**train_m, **{f"eval/{k}": v for k, v in ev.items()}}
        record = {
            "iteration": it,
            "env_steps": (it + cfg.warmup_iterations) * env_steps_per_it,
            "elapsed_s": now - t0,
            "iterations_per_s": it_s,
            "eta_s": eta,
            "checkpoint": str(ckpt),
            **last,
        }
        with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        task_m = (
            f"landing={ev['landing_success_rate']:.2f} max_rot={ev['max_backward_rotation_rad_mean']:.2f}rad "
            f"flip_fall={ev['flip_fall_rate']:.2f} "
            if "landing_success_rate" in ev
            else ""
        )
        routine_m = (
            f"flip={ev['flip_rate']:.2f} kick={ev['kick_rate']:.2f} routine={ev['routine_success_rate']:.2f} "
            if "routine_success_rate" in ev
            else ""
        )
        print(
            f"[eval] it {it:,}/{cfg.num_iterations:,} ({100.0 * it / cfg.num_iterations:5.1f}%) "
            f"return={ev['episode_return_mean']:.3f} survival={ev['survival_time_s_mean']:.2f}s "
            f"complete={ev['completion_rate']:.2f} fall={ev['fall_rate']:.2f} "
            f"track_fail={ev['tracking_fail_rate']:.2f} root_err={ev['root_pos_error_m']:.3f}m "
            f"ori_err={ev['root_orientation_error_rad']:.3f}rad joint_rmse={ev['joint_rmse_rad']:.3f}rad "
            f"{task_m}{routine_m}| wm_loss={train_m['wm/loss']:.4f} "
            f"J_img={train_m['policy/imagined_return']:.3e} v_loss={train_m['critic/loss']:.2e} "
            f"v_expl_var={train_m['critic/explained_variance']:.2f} "
            f"collect_ret={train_m['collect/episode_return']:.3f} | "
            f"elapsed={now - t0:.0f}s it/s={it_s:.2f} eta={eta:.0f}s",
            flush=True,
        )

    total = time.monotonic() - t0
    summary = {"run_dir": str(run_dir), "total_wall_s": total, "final_metrics": last}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wall={total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
