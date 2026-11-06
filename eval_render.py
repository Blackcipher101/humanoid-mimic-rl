#!/usr/bin/env python3
"""Roll out a trained policy: metrics JSON + MP4 video.

Examples:
  python eval_render.py --run-dir runs/walk-full-20261106-120000
  python eval_render.py --run-dir runs/mimic-full-... --num-episodes 128 --playback-speed 0.5
  python eval_render.py --reference-only data/generated/t1_backflip_keyframed.npz   # kinematic clip video
  python eval_render.py --run-dir runs/mimic-supermimic-full-...                      # SuperMimic checkpoint

Writes to ``<run-dir>/eval/`` (or ``--out-dir``):
  metrics.json   task-specific metrics over ``--num-episodes`` deterministic-policy episodes
  rollout_<i>.mp4 for the first ``--video-episodes`` episodes

Metrics (see ``humanoid_mimic_rl/metrics.py`` for exact definitions):
  all:       episode return, episode length, early-termination rate
  walk:      success rate, fall rate, forward-speed / planar-velocity tracking error (m/s)
             for a fixed command (``--walk-command``), pushes disabled
  getup:     success rate (standing for the final 1 s), stood-up rate, fall rate, time to stand
  backflip/mimic: max backward rotation (rad), full-rotation rate, landing success rate, fall rate
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mediapy  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from brax.training.agents.ppo import checkpoint as ppo_checkpoint  # noqa: E402
from mujoco_playground._src import mjx_env  # noqa: E402
from mujoco_playground._src.locomotion.t1 import base as t1_base  # noqa: E402

from humanoid_mimic_rl import metrics, reference, t1_model, tasks  # noqa: E402

GETUP_HOLD_S = 1.0
FLIP_ROTATION_TOLERANCE_RAD = 0.5  # landing within ~30 deg of a full turn counts
FLIP_LAND_MIN_UP_Z = 0.8
FLIP_LAND_HEIGHT_FRAC = 0.75
CAMERA = "track"  # trackcom camera on the T1 trunk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="training run directory (uses its latest checkpoint)")
    src.add_argument("--reference-only", type=Path, help="render a reference clip kinematically and exit")
    p.add_argument("--checkpoint", type=Path, default=None, help="specific checkpoint dir inside the run")
    p.add_argument("--num-episodes", type=int, default=64)
    p.add_argument("--video-episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--impl", default=None, choices=("jax", "warp"), help="default: the run's impl")
    p.add_argument("--walk-command", type=float, nargs=3, default=(0.5, 0.0, 0.0),
                   metavar=("VX_MPS", "VY_MPS", "WZ_RADPS"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--render-every", type=int, default=1, help="render every n-th control step")
    p.add_argument("--playback-speed", type=float, default=1.0, help="<1 = slow motion")
    return p.parse_args()


def render_qpos(model: mujoco.MjModel, qpos: np.ndarray, width: int, height: int) -> list[np.ndarray]:
    """Render a sequence of generalized positions with the tracking camera."""
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    frames = []
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=CAMERA)
            frames.append(renderer.render())
    finally:
        renderer.close()
    return frames


def write_video(path: Path, frames: list[np.ndarray], step_dt_s: float, render_every: int, speed: float) -> None:
    fps = speed / (step_dt_s * render_every)
    mediapy.write_video(str(path), frames, fps=fps)
    print(f"wrote {path} ({len(frames)} frames @ {fps:.1f} fps)", flush=True)


def render_reference(path: Path, args: argparse.Namespace) -> None:
    clip = reference.load_clip(path)
    mjx_env.ensure_menagerie_exists()
    scene, assets = t1_model.full_collision_flat_scene(t1_base.get_assets())
    model = mujoco.MjModel.from_xml_string(scene, assets=assets)
    out_dir = args.out_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = render_qpos(model, clip.qpos[:: args.render_every], args.width, args.height)
    write_video(out_dir / f"{path.stem}_reference.mp4", frames, 1.0 / clip.fps_hz, args.render_every,
                args.playback_speed)
    print(f"licence: {clip.licence}")


def latest_checkpoint(run_dir: Path, algo: str) -> Path:
    """Latest Brax PPO checkpoint directory, or latest SuperMimic ``.msgpack`` file."""
    entries = (run_dir / "checkpoints").iterdir()
    if algo == "supermimic":
        ckpts = sorted(p for p in entries if p.suffix == ".msgpack")
    else:
        ckpts = sorted(p for p in entries if p.is_dir())
    if not ckpts:
        raise SystemExit(f"no {algo} checkpoints in {run_dir / 'checkpoints'}")
    return ckpts[-1]


def load_policy(run_dir: Path, ckpt: Path, algo: str) -> Callable[..., Any]:
    if algo == "supermimic":
        from humanoid_mimic_rl.supermimic import trainer as sm  # pylint: disable=import-outside-toplevel

        return sm.load_policy(ckpt, run_dir / "run_config.json")
    return ppo_checkpoint.load_policy(ckpt, deterministic=True)


def make_recorder(task: str, env: Any) -> Callable[[mjx_env.State], dict[str, jax.Array]]:
    """Per-step quantities needed by the task metrics (single, un-batched state)."""

    def record(state: mjx_env.State) -> dict[str, jax.Array]:
        d = state.data
        out = {
            "qpos": d.qpos,
            "reward": state.reward,
            "done": state.done,
            "up_z": env.get_gravity(d)[2],  # playground "upvector": torso z-axis in world
            "height_m": d.qpos[2],  # trunk origin height
        }
        if task == "walk":
            out["local_linvel"] = env.get_local_linvel(d)
            out["command"] = state.info["command"]
        else:
            out["feet_contact"] = env.feet_contact(d)
        if task == "getup":
            out["standing"] = state.metrics["standing"]
        return out

    return record


def rollout(
    env: Any,
    policy: Callable[..., Any],
    task: str,
    num_episodes: int,
    steps: int,
    seed: int,
    walk_command: np.ndarray,
) -> dict[str, np.ndarray]:
    """Batched deterministic rollouts without auto-reset; returns ``(N, T, ...)`` numpy arrays."""
    record = make_recorder(task, env)
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)
    v_policy = jax.vmap(policy)
    command = jp.broadcast_to(jp.asarray(walk_command, dtype=jp.float32), (num_episodes, 3))

    def body(carry: tuple[mjx_env.State, jax.Array], _: None) -> tuple[tuple[mjx_env.State, jax.Array], Any]:
        state, key = carry
        key, sub = jax.random.split(key)
        if task == "walk":
            state.info["command"] = command
        action, _ = v_policy(state.obs, jax.random.split(sub, num_episodes))
        state = v_step(state, action)
        return (state, key), jax.vmap(record)(state)

    @jax.jit
    def run(key: jax.Array) -> Any:
        k_reset, k_act = jax.random.split(key)
        state = v_reset(jax.random.split(k_reset, num_episodes))
        _, traj = jax.lax.scan(body, (state, k_act), None, length=steps)
        return traj

    traj = run(jax.random.PRNGKey(seed))
    return {k: np.moveaxis(np.asarray(v), 0, 1) for k, v in traj.items()}


def summarize(task: str, tr: dict[str, np.ndarray], dt: float) -> dict[str, float]:
    if task == "walk":
        return metrics.summarize_walk(tr["reward"], tr["done"], tr["local_linvel"], tr["command"],
                                      tr["up_z"], tr["height_m"], dt, metrics.FallCriteria())
    if task == "getup":
        return metrics.summarize_getup(tr["reward"], tr["done"], tr["standing"], dt, GETUP_HOLD_S)
    return metrics.summarize_flip(
        tr["reward"], tr["done"], tr["qpos"][..., 3:7], tr["up_z"], tr["height_m"], tr["feet_contact"], dt,
        FLIP_ROTATION_TOLERANCE_RAD, FLIP_LAND_MIN_UP_Z, FLIP_LAND_HEIGHT_FRAC * t1_model.T1_HOME_ROOT_HEIGHT_M,
    )


def main() -> None:
    args = parse_args()
    if args.reference_only is not None:
        render_reference(args.reference_only, args)
        return

    run_cfg = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    task = run_cfg["task"]
    algo = run_cfg.get("algo", "ppo")
    impl = args.impl or run_cfg["impl"]
    ref_path = run_cfg["env_config"].get("reference_path") or None
    ckpt = args.checkpoint or latest_checkpoint(args.run_dir, algo)
    out_dir = args.out_dir or args.run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = tasks.make_env(task, impl, args.num_episodes, Path(ref_path) if ref_path else None, eval_mode=True)
    steps = tasks.episode_length(env)
    policy = load_policy(args.run_dir, ckpt, algo)
    print(f"task={task} algo={algo} checkpoint={ckpt} episodes={args.num_episodes} steps={steps}", flush=True)

    tr = rollout(env, policy, task, args.num_episodes, steps, args.seed, np.asarray(args.walk_command))
    summary = summarize(task, tr, env.dt)
    result = {
        "task": task,
        "algo": algo,
        "checkpoint": str(ckpt),
        "impl": impl,
        "seed": args.seed,
        "episode_steps": steps,
        "ctrl_dt_s": env.dt,
        "walk_command": list(args.walk_command) if task == "walk" else None,
        "metrics": summary,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    lengths = metrics.episode_lengths(tr["done"])
    for i in range(min(args.video_episodes, args.num_episodes)):
        qpos = tr["qpos"][i, : lengths[i] : args.render_every]
        frames = render_qpos(env.mj_model, qpos, args.width, args.height)
        write_video(out_dir / f"rollout_{i}.mp4", frames, env.dt, args.render_every, args.playback_speed)


if __name__ == "__main__":
    main()
