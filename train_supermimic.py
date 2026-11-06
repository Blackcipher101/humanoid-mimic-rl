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
    p.add_argument("--profile", default="full", choices=("smoke", "full"))
    p.add_argument("--impl", default="jax", choices=("jax", "warp"), help="MJX backend of the simulator env")
    p.add_argument("--iterations", type=int, default=None, help="override the profile's iteration count")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--reference", type=Path, default=None)
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
    cfg.validate()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.logdir / f"mimic-supermimic-{args.profile}-{stamp}").resolve()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    env = tasks.make_env("mimic", args.impl, cfg.num_envs, args.reference)
    eval_env = tasks.make_env("mimic", args.impl, cfg.num_eval_envs, args.reference)
    trainer = sm.SuperMimicTrainer(env, cfg)

    run_config = {
        "task": "mimic",
        "algo": "supermimic",
        "profile": args.profile,
        "impl": args.impl,
        "seed": args.seed,
        "supermimic": dataclasses.asdict(cfg),
        "env_config": env.unwrapped._config.to_dict(),  # pylint: disable=protected-access
        "versions": _versions(),
        "devices": [str(d) for d in jax.devices()],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    print(f"run dir: {run_dir}", flush=True)
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
        ev = sm.summarize_eval(traj, env.dt)
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
        print(
            f"[eval] it {it:,}/{cfg.num_iterations:,} ({100.0 * it / cfg.num_iterations:5.1f}%) "
            f"return={ev['episode_return_mean']:.3f} ep_len={ev['episode_length_s_mean']:.2f}s "
            f"landing={ev['landing_success_rate']:.2f} max_rot={ev['max_backward_rotation_rad_mean']:.2f}rad "
            f"fall={ev['fall_rate']:.2f} | wm_loss={train_m['wm/loss']:.4f} "
            f"J_img={train_m['policy/imagined_return']:.3f} v_loss={train_m['critic/loss']:.4f} "
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
