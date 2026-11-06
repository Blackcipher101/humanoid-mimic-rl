#!/usr/bin/env python3
"""CMU mocap acrobatics -> Unitree G1 reference clip, with a torque-limit feasibility report (no JAX, no GPU).

For each ``<SS>_<TT>`` trial (``cmu_mocap.TRIALS``): parse the ASF/AMC from ``--raw-dir`` (downloaded, untouched,
see ``--fetch``), retarget to the G1 (``g1_retarget``: Froude-scaled time, direction-matching IK with joint
limits, flat static foot/hand contacts, no floor penetration), then check inverse-dynamics feasibility at the
**published** peak torques (``--torque-scale 1``) with ``g1_cmu_feasibility.check_clip`` (feet and hands as
contacts; flight root wrench ~ 0; torques <= ``--torque-margin`` x peak).

If the clip fails, ONE cheap fix is tried and both results are reported: ground moves (no flight frame) are
time-scaled slower by 1.1 / 1.2 / 1.3x (first that passes, else the best); clips with flight get an exactly
ballistic centre of mass in each flight segment (``ballistic_flight_fix``).

Outputs (``--out-dir``): ``g1_cmu_<SS>_<TT>.npz`` (+ ``.report.json``) for the faithful retarget and, when a fix
changed the clip, ``g1_cmu_<SS>_<TT>_fixed.npz`` (+ report). A clip's ``passed`` is True only if it meets every
criterion at torque scale 1; ``g1_model.resolve_torque_scale`` refuses clips that failed.
Optional kinematic preview MP4 (``--preview-dir``, MuJoCo offscreen EGL).

Usage:
  python scripts/make_g1_cmu_clip.py 90_02 88_06 --g1-xml-dir ~/g1_assets/playground_g1_xmls \\
      --mesh-dir ~/g1_assets/menagerie_unitree_g1/assets --preview-dir ~/cmu_mocap/previews
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_mimic_rl import cmu_mocap, g1_backflip, g1_cmu_feasibility, g1_model, g1_retarget, reference  # noqa: E402
from humanoid_mimic_rl import flip_dynamics  # noqa: E402

TIME_SCALES = (1.1, 1.2, 1.3)  # "slower by up to 1.3x" for ground moves


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_assets(xml_dir: Path, mesh_dir: Path) -> dict[str, bytes]:
    files = [*xml_dir.glob("*.xml"), *(xml_dir / "assets").glob("*"), *mesh_dir.glob("*")]
    return {p.name: p.read_bytes() for p in files if p.is_file()}


def check(fb: flip_dynamics.FloatingBaseModel, qpos: np.ndarray, fps: float, margin: float) -> tuple[dict, bool]:
    rep = g1_cmu_feasibility.check_clip(fb, qpos, fps)
    tol_f = g1_cmu_feasibility.ROOT_FORCE_TOL_FRAC * rep.body_weight_n
    ok = rep.passes(margin, tol_f, g1_cmu_feasibility.ROOT_TORQUE_TOL_NM)
    return dataclasses.asdict(rep), ok


def compact(rep: dict, margin: float) -> dict:
    """Headline numbers; ``torque_scale_for_torque_criterion`` = peak-torque multiplier k at which the torque
    criterion alone would be met (motors: ratio scales as 1 / k); wrench criteria do not depend on k."""
    out = {"torque_scale_for_torque_criterion": rep["torque_ratio_max"] / margin}
    keys = ("torque_ratio_max", "worst_joint", "worst_frame", "flight_root_force_max_n", "flight_root_torque_max_nm",
            "contact_wrench_residual_max_n", "contact_wrench_residual_max_nm", "flight_frames", "contact_frames")  # fmt: skip
    return {**{k: rep[k] for k in keys}, **out}


def write_clip(
    rig: g1_retarget.G1Rig, fb: flip_dynamics.FloatingBaseModel, qpos: np.ndarray, fps: float, path: Path,
    source: str, report: dict,
) -> reference.ReferenceClip:  # fmt: skip
    geo = g1_cmu_feasibility.contact_geometry(fb)
    flags = g1_cmu_feasibility.contact_flags(fb, geo, qpos)
    clip = g1_retarget.to_clip(rig, qpos, fps, flags[:, :2].any(axis=1), cmu_mocap.LICENCE, source)
    reference.save_clip(path, clip)
    report["clip"] = str(path)
    report["frames"] = clip.num_frames
    report["duration_s"] = clip.duration_s
    g1_model.clip_report_path(path).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return clip


def run_trial(
    stem: str, rig: g1_retarget.G1Rig, fb: flip_dynamics.FloatingBaseModel, raw_dir: Path, out_dir: Path,
    margin: float, preview_dir: Path | None,
) -> dict:  # fmt: skip
    trial = cmu_mocap.TRIALS[stem]
    asf, amc = raw_dir / f"{trial.subject:02d}.asf", raw_dir / f"{stem}.amc"
    skeleton = cmu_mocap.parse_asf(asf)
    motion = cmu_mocap.parse_amc(amc, skeleton)
    t0 = time.monotonic()
    result = g1_retarget.retarget(rig, skeleton, motion, trial.fps_hz)
    fps = result.fps_hz
    qpos = result.qpos
    source = (
        f"CMU Graphics Lab Motion Capture Database subject {trial.subject} trial {trial.trial} "
        f"({trial.description!r}); {trial.asf_url} + {trial.amc_url}; retargeted by humanoid_mimic_rl.g1_retarget; "
        f"{g1_model.torque_scale_label(1.0)}"
    )
    base = {
        "torque_scale": 1.0,
        "model_label": g1_model.torque_scale_label(1.0),
        "trial": {"subject": trial.subject, "trial": trial.trial, "description": trial.description,
                  "amc_fps_hz": trial.fps_hz},
        "source_files": {
            "asf": {"url": trial.asf_url, "sha256": sha256(asf)},
            "amc": {"url": trial.amc_url, "sha256": sha256(amc)},
        },
        "licence_statement": cmu_mocap.LICENCE,
        "axis_and_units": (
            "AMC is Y-up; lengths x (1/0.45)*2.54/100 = 0.056444 m per unit (CMU FAQ); rotated to z-up by "
            "(x, y, z) -> (x, -z, y); frame 0 yaw-aligned to +x at the origin"
        ),
        "retarget": {
            "length_scale_g1_over_human": result.scale,
            "time_scale_froude_sqrt_scale": float(np.sqrt(result.scale)),
            "human_duration_s": (motion.num_frames - 1) / trial.fps_hz,
            "keypoint_error_cm": g1_retarget.keypoint_errors_cm(rig, qpos, result.human),
            "joint_limits": g1_retarget.joint_limit_report(rig, qpos),
            "contacts": g1_retarget.contact_quality(rig, qpos, result.human_contacts),
            "wall_s": time.monotonic() - t0,
        },
        "check_thresholds": {
            "torque_ratio_max": margin,
            "flight_root_force_n": g1_cmu_feasibility.ROOT_FORCE_TOL_FRAC * fb.mass_kg * flip_dynamics.GRAVITY_MPS2,
            "flight_root_torque_nm": g1_cmu_feasibility.ROOT_TORQUE_TOL_NM,
            "friction": g1_model.FOOT_FRICTION,
            "contact_tol_m": g1_cmu_feasibility.CONTACT_TOL_M,
        },
        "checker": "g1_cmu_feasibility.check_clip (feet+hands contact LP; best case over contact force splits)",
    }
    rep, ok = check(fb, qpos, fps, margin)
    report = {**base, "variant": "faithful retarget", "check": rep, "dynamics_check_passed": ok, "passed": ok}
    out_path = out_dir / f"g1_cmu_{stem}.npz"
    write_clip(rig, fb, qpos, fps, out_path, source, report)
    summary = {"trial": stem, "description": trial.description, "clip": str(out_path), "duration_s": (len(qpos) - 1) / fps,
               "before": compact(rep, margin), "before_passed": ok, "after": None, "after_passed": None, "fix": None}  # fmt: skip

    if not ok:
        has_flight = rep["flight_frames"] > 0
        variants = []
        if has_flight:
            fixed, n_seg = g1_cmu_feasibility.ballistic_flight_fix(fb, qpos, fps)
            variants.append((f"ballistic CoM in {n_seg} flight segment(s)", fixed, fps))
        else:
            for factor in TIME_SCALES:
                variants.append((f"time scaled {factor:g}x slower", g1_retarget.apply_time_scale(qpos, fps, factor), fps))
        best = None
        for label, q_var, f_var in variants:
            rep_v, ok_v = check(fb, q_var, f_var, margin)
            if best is None or ok_v or rep_v["torque_ratio_max"] < best[2]["torque_ratio_max"]:
                best = (label, q_var, rep_v, ok_v)
            if ok_v:
                break
        label, q_fix, rep_fix, ok_fix = best
        report_fix = {**base, "variant": label, "check": rep_fix, "dynamics_check_passed": ok_fix, "passed": ok_fix,
                      "unfixed_report": str(g1_model.clip_report_path(out_path))}  # fmt: skip
        fix_path = out_dir / f"g1_cmu_{stem}_fixed.npz"
        write_clip(rig, fb, q_fix, fps, fix_path, source + f"; fix: {label}", report_fix)
        summary.update(after=compact(rep_fix, margin), after_passed=ok_fix, fix=label, fixed_clip=str(fix_path))
        out_path_preview = fix_path if ok_fix else out_path
    else:
        out_path_preview = out_path
    if preview_dir is not None:
        summary["preview"] = render_preview(rig, reference.load_clip(out_path), preview_dir / f"g1_cmu_{stem}.mp4")
        if out_path_preview != out_path:
            summary["preview_fixed"] = render_preview(
                rig, reference.load_clip(out_path_preview), preview_dir / f"g1_cmu_{stem}_fixed.mp4"
            )
    return summary


def render_preview(rig: g1_retarget.G1Rig, clip: reference.ReferenceClip, path: Path) -> str | None:
    """Kinematic preview MP4 (no dynamics): the clip's qpos rendered offscreen. Returns the path or ``None``."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import imageio.v2 as imageio  # noqa: PLC0415
        import mujoco  # noqa: PLC0415

        path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(rig.m, 480, 640)
        data = mujoco.MjData(rig.m)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 3.2, -12.0, 150.0
        with imageio.get_writer(str(path), fps=int(clip.fps_hz), codec="libx264", macro_block_size=1) as w:
            for q in clip.qpos:
                data.qpos[:] = q
                mujoco.mj_forward(rig.m, data)
                cam.lookat[:] = (q[0], q[1], 0.6)
                renderer.update_scene(data, cam)
                w.append_data(renderer.render())
        renderer.close()
    except Exception as exc:  # noqa: BLE001  (preview is optional; report why it was skipped)
        print(f"preview skipped for {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return str(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trials", nargs="+", help="trial stems, e.g. 90_02 88_06")
    p.add_argument("--raw-dir", type=Path, default=Path.home() / "cmu_mocap" / "raw")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "generated")
    p.add_argument("--preview-dir", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None, help="default: <out-dir>/g1_cmu_summary.json")
    p.add_argument("--torque-margin", type=float, default=g1_cmu_feasibility.TORQUE_MARGIN)
    p.add_argument("--torque-scale", type=float, default=1.0, help="must stay 1: the question is the real robot")
    p.add_argument("--g1-xml-dir", type=Path, required=True)
    p.add_argument("--mesh-dir", type=Path, required=True)
    args = p.parse_args()
    if args.torque_scale != 1.0:
        raise SystemExit("this tool only checks the published torque limits (--torque-scale 1)")
    unknown = [t for t in args.trials if t not in cmu_mocap.TRIALS]
    if unknown:
        raise SystemExit(f"unknown trials {unknown}; known: {sorted(cmu_mocap.TRIALS)}")
    assets = load_assets(args.g1_xml_dir, args.mesh_dir)
    rig = g1_retarget.G1Rig(assets)
    fb = g1_backflip.floating_base(rig.m)
    summaries = []
    for stem in args.trials:
        s = run_trial(stem, rig, fb, args.raw_dir, args.out_dir, args.torque_margin, args.preview_dir)
        summaries.append(s)
        print(json.dumps(s, indent=1, allow_nan=False), flush=True)
    (args.summary or args.out_dir / "g1_cmu_summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
