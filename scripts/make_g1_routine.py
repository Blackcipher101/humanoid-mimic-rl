#!/usr/bin/env python3
"""Join the CMU 87/3 backflip and the CMU 88/6 jump + spin kick into one G1 "routine" clip (no JAX, no GPU).

Backflip -> 1 s transition through the G1 stand (feet planted, IK foot locks) -> spin kick -> stand hold; see
``humanoid_mimic_rl.g1_routine``. Writes ``g1_routine_backflip_spinkick.npz`` + ``.report.json`` (sources and
CMU licence statement, segment / milestone windows for ``humanoid_mimic_rl.routine``, continuity numbers and the
torque feasibility check of the whole routine at the published G1 torques, ``passed`` = that check) and an
optional kinematic preview MP4. The routine FAILS the torque check at scale 1 (accepted: train with
``--allow-infeasible-clip``).

Usage:
  python scripts/make_g1_routine.py --g1-xml-dir ~/g1_assets/playground_g1_xmls \\
      --mesh-dir ~/g1_assets/menagerie_unitree_g1/assets --preview ~/cmu_mocap/previews/g1_routine_backflip_spinkick.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_g1_cmu_clip as cmu_tool  # noqa: E402

from humanoid_mimic_rl import cmu_mocap, g1_backflip, g1_cmu_feasibility, g1_model, g1_retarget, g1_routine  # noqa: E402
from humanoid_mimic_rl import reference, routine  # noqa: E402

FLIP_SOURCE = "g1_cmu_87_03_fixed.npz"
KICK_SOURCE = "g1_cmu_88_06_fixed.npz"
OUT_NAME = "g1_routine_backflip_spinkick.npz"


def routine_section(clip: reference.ReferenceClip, r: g1_routine.Routine) -> dict[str, object]:
    """Milestone windows and kick description for ``routine.windows_from_report`` plus reference rotations."""
    kick_w = r.kick_window_s
    t = np.arange(clip.num_frames) / clip.fps_hz
    in_kick = (t >= kick_w[0]) & (t <= kick_w[1])
    peaks = clip.site_pos[in_kick, :2, 2].max(axis=0)
    foot = int(np.argmax(peaks))
    _, d_yaw = routine.rotation_increments(clip.qpos[:-1, 3:7], clip.qpos[1:, 3:7], np)
    t1 = t[1:]
    yaw = float(np.sum(d_yaw[(t1 > kick_w[0]) & (t1 <= kick_w[1])]))
    sign = 1.0 if yaw >= 0.0 else -1.0
    win = routine.RoutineWindows(*r.flip_window_s, *kick_w, foot, float(peaks[foot]), sign)
    prog = routine.reference_progress(clip.qpos, clip.site_pos[:, foot, 2], clip.fps_hz, routine.RoutineThresholds(), win)
    return {
        "flip_window_s": list(r.flip_window_s),
        "kick_window_s": list(r.kick_window_s),
        "kick_foot_site": clip.site_names[foot],
        "kick_foot_peak_height_m": float(peaks[foot]),
        "kick_yaw_sign": sign,
        "reference_flip_backward_rotation_deg": float(np.degrees(prog.flip_rot_rad.max())),
        "reference_kick_yaw_deg": float(np.degrees(prog.kick_yaw_rad.max() )),
        "window_definitions": (
            "flip: from the peak forward lean before take-off to 0.2 s after touch-down (rotation counted from "
            "there; a window limited to the flight phase holds only ~225 deg of this clip); kick: from 0.3 s "
            "before the first take-off to the last touch-down. Times are seconds from the first frame."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "generated")
    p.add_argument("--preview", type=Path, default=None)
    p.add_argument("--g1-xml-dir", type=Path, required=True)
    p.add_argument("--mesh-dir", type=Path, required=True)
    p.add_argument("--torque-margin", type=float, default=g1_cmu_feasibility.TORQUE_MARGIN)
    p.add_argument("--torque-scale", type=float, default=1.0,
                   help="peak-torque multiplier recorded in the report and used by the torque check; != 1 is a MODIFIED "
                        "robot (output gets a _ts<k> suffix and the torque_scale_label disclaimer)")
    args = p.parse_args()
    if not args.torque_scale > 0.0:
        p.error(f"--torque-scale must be positive, got {args.torque_scale}")
    scale = float(args.torque_scale)
    flip_path, kick_path = args.data_dir / FLIP_SOURCE, args.data_dir / KICK_SOURCE
    flip, kick = reference.load_clip(flip_path), reference.load_clip(kick_path)
    rig = g1_retarget.G1Rig(cmu_tool.load_assets(args.g1_xml_dir, args.mesh_dir))
    fb = g1_backflip.floating_base(rig.m)

    r = g1_routine.assemble(rig, flip, kick)
    out_path = args.data_dir / (OUT_NAME if scale == 1.0 else OUT_NAME.replace(".npz", f"_ts{scale:g}.npz"))
    source = (
        "CMU Graphics Lab Motion Capture Database subjects 87 trial 3 (backflip) and 88 trial 6 (jump and spin kick), "
        "http://mocap.cs.cmu.edu; retargeted to the G1 by humanoid_mimic_rl.g1_retarget, joined by "
        f"humanoid_mimic_rl.g1_routine; {g1_model.torque_scale_label(scale)}"
    )
    # motor torque ratios scale as 1 / k, so the torque criterion at scale k is ratio <= margin * k
    rep, ok = cmu_tool.check(fb, r.qpos, r.fps_hz, args.torque_margin * scale)
    src_reports = {n: json.loads(g1_model.clip_report_path(pth).read_text(encoding="utf-8"))
                   for n, pth in (("backflip", flip_path), ("spin_kick", kick_path))}
    report = {
        "torque_scale": scale,
        "model_label": g1_model.torque_scale_label(scale),
        "variant": "routine: CMU 87/3 backflip -> transition -> CMU 88/6 spin kick -> stand hold",
        "sources": {n: {"clip": str(pth), "trial": src_reports[n]["trial"], "source_files": src_reports[n]["source_files"],
                        "source_variant": src_reports[n]["variant"], "trim": r.sources[n]}
                    for n, pth in (("backflip", flip_path), ("spin_kick", kick_path))},
        "licence_statement": cmu_mocap.LICENCE,
        "redistribution_note": (
            "CMU: 'you may not resell this data directly, even in converted form'. This routine is a converted "
            "derivative: do not redistribute it (or the source clips) as a dataset or product."
        ),
        "segments_s": {k: list(v) for k, v in r.segments.items()},
        "routine": None,
        "continuity": g1_routine.continuity_report(rig, r),
        "check": rep,
        "check_thresholds": {"torque_ratio_max": args.torque_margin * scale},
        "dynamics_check_passed": ok,
        "passed": ok,
        "passed_note": "accepted when False: train with --allow-infeasible-clip (the robot keeps the torque limits recorded here)",
        "checker": "g1_cmu_feasibility.check_clip (feet+hands contact LP; best case over contact force splits)",
    }
    report["torque_scale_for_torque_criterion"] = rep["torque_ratio_max"] / args.torque_margin
    # write once to get the clip (site positions), then add the routine section that needs them
    clip = cmu_tool.write_clip(rig, fb, r.qpos, r.fps_hz, out_path, source, report)
    report["routine"] = routine_section(clip, r)
    g1_model.clip_report_path(out_path).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("segments_s", "routine", "continuity", "passed")}, indent=1))
    print(f"torque_ratio_max={rep['torque_ratio_max']:.2f} worst={rep['worst_joint']} duration={clip.duration_s:.2f}s")
    if args.preview is not None:
        print("preview:", cmu_tool.render_preview(rig, clip, args.preview))
    return 0


if __name__ == "__main__":
    sys.exit(main())
