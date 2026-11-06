#!/usr/bin/env python3
"""Generate the keyframed T1 backflip reference clip used by the ``mimic`` task.

The clip is produced procedurally from the keyframes in
``humanoid_mimic_rl/keyframes.py`` (authored for this repository, no
third-party motion data) and fitted to the floor / a ballistic CoM path with
MuJoCo forward kinematics (``humanoid_mimic_rl/clip_builder.py``).

Model source:
  * default: the T1 assets of the installed ``mujoco_playground`` package
    (meshes from its pinned mujoco_menagerie checkout, downloaded on first use);
  * ``--t1-xml-dir DIR --mesh-dir DIR``: explicit directories holding
    playground's ``t1/xmls`` files and menagerie's ``booster_t1/assets`` meshes
    (lets the clip be built without JAX installed).

Usage:
  python scripts/make_reference.py [--out data/generated/t1_backflip_keyframed.npz]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_mimic_rl import clip_builder, keyframes, reference, t1_model  # noqa: E402
from humanoid_mimic_rl.mj_kinematics import MjKinematics  # noqa: E402

LICENCE = (
    "Procedurally generated from keyframes authored in humanoid_mimic_rl/keyframes.py; "
    "contains no third-party motion data. Same licence as this repository."
)
SOURCE = "humanoid_mimic_rl.keyframes.t1_backflip_keyframes + clip_builder.build_clip"


def load_assets(t1_xml_dir: Path | None, mesh_dir: Path | None) -> dict[str, bytes]:
    """T1 asset dict keyed by basename, as ``t1.base.get_assets()`` builds it."""
    if t1_xml_dir is None and mesh_dir is None:
        from mujoco_playground._src import mjx_env  # pylint: disable=import-outside-toplevel
        from mujoco_playground._src.locomotion.t1 import base as t1_base  # pylint: disable=import-outside-toplevel

        mjx_env.ensure_menagerie_exists()
        return t1_base.get_assets()
    if t1_xml_dir is None or mesh_dir is None:
        raise SystemExit("--t1-xml-dir and --mesh-dir must be given together")
    assets: dict[str, bytes] = {}
    for path in [*t1_xml_dir.glob("*.xml"), *mesh_dir.glob("*.stl")]:
        assets[path.name] = path.read_bytes()
    return assets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "generated" / "t1_backflip_keyframed.npz")
    parser.add_argument("--rate-hz", type=float, default=100.0, help="clip sample rate (Hz)")
    parser.add_argument(
        "--landing-offset-x-m", type=float, default=-0.3,
        help="landing foot x relative to take-off (m, negative = backwards)",
    )
    parser.add_argument(
        "--auto-landing-offset", action="store_true",
        help="instead choose the landing x so flight vx equals the stance CoM vx at take-off"
    )
    parser.add_argument("--t1-xml-dir", type=Path, default=None)
    parser.add_argument("--mesh-dir", type=Path, default=None)
    args = parser.parse_args()

    scene_xml, assets = t1_model.full_collision_flat_scene(load_assets(args.t1_xml_dir, args.mesh_dir))
    model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
    joint_names = tuple(model.joint(i).name for i in range(1, model.njnt))
    if joint_names != t1_model.T1_JOINT_NAMES:
        raise SystemExit(f"T1 joint order changed: {joint_names}")

    keys = keyframes.t1_backflip_keyframes()
    keyframes.validate_keyframes(keys, joint_names, model.jnt_range[1:])
    traj = keyframes.interpolate_keyframes(keys, joint_names, model.keyframe("home").qpos[7:], args.rate_hz)
    kin = MjKinematics(model, "Trunk", t1_model.T1_FOOT_SPHERE_GEOMS, t1_model.T1_END_EFFECTOR_SITES)
    diag, clip = clip_builder.build_clip(
        traj, keyframes.flight_phase(keys), kin, joint_names,
        None if args.auto_landing_offset else args.landing_offset_x_m, LICENCE, SOURCE
    )
    reference.save_clip(args.out, clip)
    print(json.dumps({"out": str(args.out), "frames": clip.num_frames, "duration_s": clip.duration_s,
                      **dataclasses.asdict(diag)}, indent=2))
    if diag.min_foot_height_m < -1e-3:
        print(f"WARNING: feet penetrate the floor by {-diag.min_foot_height_m:.3f} m during flight",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
