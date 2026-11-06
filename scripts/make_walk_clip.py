#!/usr/bin/env python3
"""Reference clip from the trained T1 walk policy (physically feasible by construction).

Rolls the PPO walk policy out in playground's ``T1JoystickFlatTerrain`` env
(pushes off, fixed velocity command), keeps the most upright episode that
never terminated, cuts a steady window after ``--warmup-s`` and stores the
**simulated** ``qpos``/``qvel`` (MuJoCo conventions; no finite differences) at
the control rate. The clip is rotated so the window starts at the origin
heading +x, end-effector sites are computed with MuJoCo FK on the
full-collision T1 used by the ``mimic`` task (same bodies, joints and inertias
as the walk env; only extra floor colliders), and per-frame foot contact is
"any foot contact sphere within ``CONTACT_TOL_M`` of the floor".

A sanity inverse-dynamics check is printed: per frame, the floating-base rows
of ``qfrc_inverse`` (constraints disabled) are balanced by a minimum-norm
least-squares wrench on the feet in contact. Because the split between two
feet is not unique this is approximate; the clip itself comes from simulation.

Usage (ws1, needs JAX + playground):
  python scripts/make_walk_clip.py --run-dir runs/walk-full-<stamp> [--out data/generated/t1_walk_from_policy.npz]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import eval_render  # noqa: E402  (sets MUJOCO_GL / XLA env vars before JAX is imported)
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco_playground._src.locomotion.t1 import base as t1_base  # noqa: E402

from humanoid_mimic_rl import clip_tools, reference, t1_model, tasks  # noqa: E402
from humanoid_mimic_rl.clip_tools import CONTACT_TOL_M  # noqa: E402
LICENCE = (
    "Recorded from a PPO walk policy trained in this repository on the Booster T1 model "
    "(MuJoCo Playground / Menagerie, Apache-2.0); no third-party motion data. Same licence as this repository."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="walk PPO run directory")
    p.add_argument("--checkpoint", type=Path, default=None, help="default: latest in the run")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "generated" / "t1_walk_from_policy.npz")
    p.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX_MPS", "VY_MPS", "WZ_RADPS"))
    p.add_argument("--warmup-s", type=float, default=3.0, help="discard this much from the start (s)")
    p.add_argument("--duration-s", type=float, default=8.0, help="clip length (s)")
    p.add_argument("--episodes", type=int, default=16, help="rollouts to choose from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--impl", default="jax", choices=("jax", "warp"))
    return p.parse_args()


def simulate(args: argparse.Namespace, total_s: float) -> tuple[dict[str, np.ndarray], float]:
    """Batched deterministic rollouts of ``total_s`` seconds; returns ``(N, T, ...)`` arrays and the control dt."""
    env = tasks.make_env("walk", args.impl, args.episodes, eval_mode=True)
    steps = int(round(total_s / env.dt)) + 1
    policy = eval_render.load_policy(args.run_dir, args.checkpoint or eval_render.latest_checkpoint(args.run_dir, "ppo"), "ppo")
    n = args.episodes
    command = jp.broadcast_to(jp.asarray(args.command, dtype=jp.float32), (n, 3))
    v_reset, v_step, v_policy = jax.vmap(env.reset), jax.vmap(env.step), jax.vmap(policy)

    def body(carry, _):
        state, key = carry
        key, sub = jax.random.split(key)
        state.info["command"] = command
        action, _ = v_policy(state.obs, jax.random.split(sub, n))
        state = v_step(state, action)
        rec = {"qpos": state.data.qpos, "qvel": state.data.qvel, "done": state.done,
               "up_z": jax.vmap(lambda d: env.get_gravity(d)[2])(state.data)}
        return (state, key), rec

    @jax.jit
    def run(key):
        k_reset, k_act = jax.random.split(key)
        state = v_reset(jax.random.split(k_reset, n))
        return jax.lax.scan(body, (state, k_act), None, length=steps)[1]

    traj = run(jax.random.PRNGKey(args.seed))
    return {k: np.moveaxis(np.asarray(v), 0, 1) for k, v in traj.items()}, float(env.dt)


def main() -> int:
    args = parse_args()
    traj, dt = simulate(args, args.warmup_s + args.duration_s)
    fps_steps = int(round(args.duration_s / dt)) + 1
    start = int(round(args.warmup_s / dt))
    ok = (~np.any(traj["done"] > 0.5, axis=1)) & (np.min(traj["up_z"], axis=1) > 0.8)
    if not np.any(ok):
        raise SystemExit("no rollout walked the whole window without falling; check the policy/command")
    best = int(np.argmax(np.where(ok, np.min(traj["up_z"], axis=1), -np.inf)))
    qpos, qvel = clip_tools.yaw_align(traj["qpos"][best, start : start + fps_steps], traj["qvel"][best, start : start + fps_steps])

    scene, assets = t1_model.full_collision_flat_scene(t1_base.get_assets())
    model = mujoco.MjModel.from_xml_string(scene, assets=assets)
    data = mujoco.MjData(model)
    site_ids = [model.site(s).id for s in t1_model.T1_END_EFFECTOR_SITES]
    spheres = [model.geom(g).id for g in t1_model.T1_FOOT_SPHERE_GEOMS]
    sites, contact = [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        sites.append(data.site_xpos[site_ids].copy())
        contact.append(bool(np.min(data.geom_xpos[spheres, 2] - model.geom_size[spheres, 0]) < CONTACT_TOL_M))
    clip = reference.ReferenceClip(
        fps_hz=1.0 / dt,
        qpos=qpos,
        qvel=qvel,
        joint_names=t1_model.T1_JOINT_NAMES,
        site_names=t1_model.T1_END_EFFECTOR_SITES,
        site_pos=np.array(sites),
        licence=LICENCE,
        source=f"scripts/make_walk_clip.py: {args.run_dir} episode {best}, command {list(args.command)}, "
        f"window {args.warmup_s}-{args.warmup_s + args.duration_s} s",
        foot_contact=np.array(contact),
    )
    reference.save_clip(args.out, clip)

    speed = (qpos[-1, 0] - qpos[0, 0]) / clip.duration_s
    summary = {
        "clip": str(args.out),
        "frames": clip.num_frames,
        "fps_hz": clip.fps_hz,
        "duration_s": clip.duration_s,
        "episode": best,
        "mean_forward_speed_mps": float(speed),
        "commanded_mps": list(args.command),
        "min_up_z": float(np.min(traj["up_z"][best, start : start + fps_steps])),
        "frames_with_foot_contact_frac": float(np.mean(contact)),
        "sanity_inverse_dynamics": clip_tools.sanity_inverse_dynamics(model, qpos, qvel, clip.fps_hz),
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
