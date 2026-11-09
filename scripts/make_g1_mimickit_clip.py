#!/usr/bin/env python3
"""MimicKit humanoid motion -> Unitree G1 reference clip, with the torque-limit feasibility report (no JAX, no GPU).

Same pipeline as ``make_g1_cmu_clip.py`` with a MimicKit source: the humanoid motion (``mimickit_motion``:
restricted unpickler, MuJoCo forward kinematics of the MimicKit humanoid MJCF, key points spline-resampled to
120 Hz) is retargeted by ``g1_retarget.retarget_human`` and checked with ``g1_cmu_feasibility.check_clip`` at the
published peak torques. Clips with flight also get the ballistic-CoM fix (``*_fixed.npz``). The report records
``torque_scale_for_torque_criterion`` (the peak-torque multiplier at which the torque criterion alone would pass).

MimicKit's motion data has no stated licence: the converted clip is for local training only.

Usage:
  python scripts/make_g1_mimickit_clip.py --motion ~/ext_mimickit_data/x/MimicKit_Data/motions/humanoid/humanoid_backflip.pkl \\
      --humanoid-xml ~/ext_mimickit_data/x/MimicKit_Data/assets/humanoid/humanoid.xml \\
      --g1-xml-dir ~/g1_assets/playground_g1_xmls --mesh-dir ~/g1_assets/menagerie_unitree_g1/assets \\
      --name backflip --preview-dir previews
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_g1_cmu_clip as cmu  # noqa: E402

from humanoid_mimic_rl import g1_backflip, g1_cmu_feasibility, g1_model, g1_retarget, mimickit_motion, reference  # noqa: E402


def write_clip(rig: g1_retarget.G1Rig, fb, qpos: np.ndarray, fps: float, path: Path, source: str,  # noqa: ANN001
               report: dict) -> None:  # fmt: skip
    geo = g1_cmu_feasibility.contact_geometry(fb)
    flags = g1_cmu_feasibility.contact_flags(fb, geo, qpos)
    clip = g1_retarget.to_clip(rig, qpos, fps, flags[:, :2].any(axis=1), mimickit_motion.LICENCE, source)
    reference.save_clip(path, clip)
    report.update(clip=str(path), frames=clip.num_frames, duration_s=clip.duration_s)
    g1_model.clip_report_path(path).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", type=Path, required=True, help="MimicKit humanoid motion .pkl")
    p.add_argument("--humanoid-xml", type=Path, required=True, help="MimicKit assets/humanoid/humanoid.xml")
    p.add_argument("--name", required=True, help="clip name: writes g1_mimickit_<name>.npz")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "generated")
    p.add_argument("--preview-dir", type=Path, default=None)
    p.add_argument("--torque-margin", type=float, default=g1_cmu_feasibility.TORQUE_MARGIN)
    p.add_argument("--torque-scale", type=float, default=1.0,
                   help="peak-torque multiplier k to check at (k != 1 is a MODIFIED robot: output gets a _ts<k> suffix "
                        "and the torque_scale_label disclaimer; motor torque ratios scale as 1 / k)")
    p.add_argument("--g1-xml-dir", type=Path, required=True)
    p.add_argument("--mesh-dir", type=Path, required=True)
    args = p.parse_args()
    if not args.torque_scale > 0.0:
        p.error(f"--torque-scale must be positive, got {args.torque_scale}")
    scale = float(args.torque_scale)
    margin = args.torque_margin * scale  # torque criterion at scale k: ratio (at k = 1) <= margin * k
    suffix = "" if scale == 1.0 else f"_ts{scale:g}"

    rig = g1_retarget.G1Rig(cmu.load_assets(args.g1_xml_dir, args.mesh_dir))
    fb = g1_backflip.floating_base(rig.m)
    fk = mimickit_motion.HumanoidFK(args.humanoid_xml)
    fps_in, frames = mimickit_motion.load_motion(args.motion)
    t0 = time.monotonic()
    human = mimickit_motion.human_keypoints(fk, fps_in, frames)
    result = g1_retarget.retarget_human(rig, human, log=print)
    qpos, fps = result.qpos, result.fps_hz
    source = (f"MimicKit motion {args.motion.name} ({frames.shape[0]} frames @ {fps_in:g} fps); retargeted by "
              f"humanoid_mimic_rl.g1_retarget; {g1_model.torque_scale_label(scale)}")  # fmt: skip
    base = {
        "torque_scale": scale,
        "model_label": g1_model.torque_scale_label(scale),
        "check_thresholds": {"torque_ratio_max": margin},
        "source_files": {"motion": {"path": str(args.motion), "sha256": cmu.sha256(args.motion)},
                         "humanoid_xml": {"path": str(args.humanoid_xml), "sha256": cmu.sha256(args.humanoid_xml)}},
        "licence_statement": mimickit_motion.LICENCE,
        "redistribution_note": "MimicKit motion data has no stated licence: do not redistribute this clip.",
        "retarget": {
            "length_scale_g1_over_human": result.scale,
            "time_scale_froude_sqrt_scale": float(np.sqrt(result.scale)),
            "human_duration_s": (frames.shape[0] - 1) / fps_in,
            "keypoint_error_cm": g1_retarget.keypoint_errors_cm(rig, qpos, result.human),
            "joint_limits": g1_retarget.joint_limit_report(rig, qpos),
            "contacts": g1_retarget.contact_quality(rig, qpos, result.human_contacts),
            "wall_s": time.monotonic() - t0,
        },
        "checker": "g1_cmu_feasibility.check_clip (feet+hands contact LP; best case over contact force splits)",
    }
    rep, ok = cmu.check(fb, qpos, fps, margin)
    out = args.out_dir / f"g1_mimickit_{args.name}{suffix}.npz"
    write_clip(rig, fb, qpos, fps, out, source,
               {**base, "variant": "faithful retarget", "check": rep, "dynamics_check_passed": ok, "passed": ok})
    summary = {"clip": str(out), "duration_s": (len(qpos) - 1) / fps, "before": cmu.compact(rep, args.torque_margin),
               "before_passed": ok}  # fmt: skip
    if not ok and rep["flight_frames"] > 0:
        fixed, n_seg = g1_cmu_feasibility.ballistic_flight_fix(fb, qpos, fps)
        rep_f, ok_f = cmu.check(fb, fixed, fps, margin)
        label = f"ballistic CoM in {n_seg} flight segment(s)"
        fix_out = args.out_dir / f"g1_mimickit_{args.name}{suffix}_fixed.npz"
        write_clip(rig, fb, fixed, fps, fix_out, source + f"; fix: {label}",
                   {**base, "variant": label, "check": rep_f, "dynamics_check_passed": ok_f, "passed": ok_f,
                    "unfixed_report": str(g1_model.clip_report_path(out))})  # fmt: skip
        summary.update(fixed_clip=str(fix_out), after=cmu.compact(rep_f, args.torque_margin), after_passed=ok_f)
    if args.preview_dir is not None:
        summary["preview"] = cmu.render_preview(rig, reference.load_clip(out), args.preview_dir / f"{out.stem}.mp4")
    print(json.dumps(summary, indent=1, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
