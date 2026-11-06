#!/usr/bin/env python3
"""Train a Brax PPO policy for one task on MJX.

Examples:
  python train.py --task walk --profile smoke
  python train.py --task mimic --profile full --reference data/generated/t1_backflip_keyframed.npz

Outputs go to ``<logdir>/<task>-<profile>-<timestamp>/``:
  run_config.json   task, PPO and env config, package versions, devices
  progress.jsonl    one JSON line per evaluation (all Brax metrics + timing)
  checkpoints/      Brax PPO checkpoints (one directory per evaluation)
  summary.json      wall-clock timing and final evaluation metrics
One line per evaluation is also printed (flushed) for ``tee``.
"""

from __future__ import annotations

import os

# Must be set before JAX / MuJoCo initialise.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import functools  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from mujoco_playground import registry, wrapper  # noqa: E402

from humanoid_mimic_rl import configs, tasks  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
_PACKAGES = ("jax", "jaxlib", "mujoco", "mujoco-mjx", "playground", "brax", "warp-lang")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(configs.TASKS))
    p.add_argument("--profile", default="full", choices=("smoke", "full"))
    p.add_argument("--impl", default="jax", choices=("jax", "warp"), help="MJX implementation")
    p.add_argument("--num-timesteps", type=int, default=None, help="override the profile's env-step budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--reference", "--clip", dest="reference", type=Path, default=None,
                   help="reference clip (.npz) for mimic tasks")
    p.add_argument("--torque-scale", type=float, default=None,
                   help="G1 flip tasks: peak-torque multiplier; must match the clip report (default: read from it)")
    p.add_argument("--hold-final-s", type=float, default=None,
                   help="mimic tasks: hold the last frame this long after the clip (default 0.6 s; 0 for walking)")
    p.add_argument("--termination", default=None, choices=("root_pose", "height_tilt"),
                   help="mimic tasks: root_pose (default: 3D root drift / full orientation) or height_tilt "
                        "(SuperMimic App. A: root height and tilt error only)")
    return p.parse_args()


def _versions() -> dict[str, str]:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def _to_float(x: Any) -> float:
    return float(np.asarray(x))


class ProgressLogger:
    """Prints one line per evaluation and appends all metrics to a JSONL file."""

    def __init__(self, path: Path, total_steps: int) -> None:
        self._path = path
        self._total = total_steps
        self._t_start = time.monotonic()
        self._t_first: float | None = None
        self._steps_first = 0
        self.last_metrics: dict[str, float] = {}

    @property
    def compile_and_first_eval_s(self) -> float | None:
        return None if self._t_first is None else self._t_first - self._t_start

    def __call__(self, num_steps: int, metrics: dict[str, Any]) -> None:
        now = time.monotonic()
        if self._t_first is None:
            self._t_first, self._steps_first = now, num_steps
        elapsed = now - self._t_start
        train_s = now - self._t_first
        sps = (num_steps - self._steps_first) / train_s if train_s > 0 else float("nan")
        eta = (self._total - num_steps) / sps if sps > 0 else float("nan")
        flat = {k: _to_float(v) for k, v in metrics.items()}
        self.last_metrics = flat
        record = {"num_steps": int(num_steps), "elapsed_s": elapsed, "env_steps_per_s": sps, "eta_s": eta, **flat}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        extras = " ".join(
            f"{k.removeprefix('eval/episode_')}={v:.3f}"
            for k, v in flat.items()
            if k.startswith("eval/episode_") and not k.startswith("eval/episode_reward/") and "_std" not in k
            and k != "eval/episode_reward"
        )
        print(
            f"[eval] step {num_steps:,}/{self._total:,} ({100.0 * num_steps / self._total:5.1f}%) "
            f"reward={flat.get('eval/episode_reward', float('nan')):.3f} "
            f"ep_len={flat.get('eval/avg_episode_length', float('nan')):.1f} "
            f"elapsed={elapsed:.0f}s sps={sps:,.0f} eta={eta:.0f}s {extras}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    spec = configs.get_task(args.task)
    ppo_cfg = spec.profile(args.profile)
    if args.num_timesteps is not None:
        ppo_cfg = dataclasses.replace(ppo_cfg, num_timesteps=args.num_timesteps)
    ppo_cfg.validate()

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.logdir / f"{args.task}-{args.profile}-{stamp}").resolve()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    env = tasks.make_env(args.task, args.impl, ppo_cfg.num_envs, args.reference, hold_final_s=args.hold_final_s,
                         torque_scale=args.torque_scale, termination=args.termination)
    eval_env = tasks.make_env(args.task, args.impl, ppo_cfg.num_eval_envs, args.reference, hold_final_s=args.hold_final_s,
                         torque_scale=args.torque_scale, termination=args.termination,
                         eval_mode=args.task in ("mimic", "g1_mimic"))  # walk eval is unchanged
    episode_length = tasks.episode_length(env)

    run_config = {
        "task": args.task,
        "profile": args.profile,
        "impl": args.impl,
        "seed": args.seed,
        "episode_length": episode_length,
        "termination": getattr(env.unwrapped._config, "termination", None),  # pylint: disable=protected-access
        "ppo": dataclasses.asdict(ppo_cfg),
        "env_config": env.unwrapped._config.to_dict(),  # pylint: disable=protected-access
        "model_label": getattr(env.unwrapped, "model_label", None),
        "domain_randomization": spec.domain_randomization,
        "versions": _versions(),
        "devices": [str(d) for d in jax.devices()],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    print(f"run dir: {run_dir}", flush=True)
    if run_config["model_label"]:
        print(f"model: {run_config['model_label']}", flush=True)
    print(f"devices: {run_config['devices']}  versions: {run_config['versions']}", flush=True)
    print(
        f"task={args.task} profile={args.profile} impl={args.impl} num_timesteps={ppo_cfg.num_timesteps:,} "
        f"num_envs={ppo_cfg.num_envs} episode_length={episode_length} "
        f"env_steps_per_update={ppo_cfg.env_steps_per_update:,}",
        flush=True,
    )

    net = ppo_cfg.network
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=net.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=net.value_hidden_layer_sizes,
        policy_obs_key=net.policy_obs_key,
        value_obs_key=net.value_obs_key,
    )
    randomization_fn = (
        registry.get_domain_randomizer(tasks.WALK_ENV_NAME) if spec.domain_randomization else None
    )
    logger = ProgressLogger(run_dir / "progress.jsonl", ppo_cfg.num_timesteps)

    t0 = time.monotonic()
    ppo.train(
        environment=env,
        eval_env=eval_env,
        episode_length=episode_length,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomization_fn,
        network_factory=network_factory,
        seed=args.seed,
        save_checkpoint_path=str(ckpt_dir),
        progress_fn=logger,
        **ppo_cfg.brax_kwargs(),
    )
    total_s = time.monotonic() - t0

    checkpoints = sorted(p for p in ckpt_dir.iterdir() if p.is_dir())
    summary = {
        "run_dir": str(run_dir),
        "total_wall_s": total_s,
        "compile_and_first_eval_s": logger.compile_and_first_eval_s,
        "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "final_metrics": logger.last_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wall={total_s:.0f}s latest_checkpoint={summary['latest_checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
