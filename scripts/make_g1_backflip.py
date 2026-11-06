#!/usr/bin/env python3
"""Build the dynamically consistent G1 backflip reference clip and its feasibility report.

Default: rebuild the clip from the optimised parameters in
``humanoid_mimic_rl/data/g1_backflip_solution.json`` (seconds). ``--optimize``
runs the trajectory optimisation (tens of minutes to hours, ``--workers``
processes; local test: ~1.6 s per CMA-ES generation with 23 workers) and
writes that file. NOTE: no feasible solution has been found yet (see README);
the file only exists after a successful ``--optimize`` run.

The clip is then checked by ``flip_dynamics.check_clip``, an inverse-dynamics
test that only sees the clip (finite-difference velocities/accelerations):
in-flight root wrench, foot-contact feasibility in stance, joint torques vs
actuator caps. The script exits with status 1 if a check fails.

Model source: installed ``mujoco_playground`` (default) or
``--g1-xml-dir`` (playground's ``locomotion/g1/xmls``) + ``--mesh-dir``
(menagerie ``unitree_g1/assets``), so it also runs without JAX.

Usage:
  python scripts/make_g1_backflip.py [--optimize --workers 16] [--out data/generated/g1_backflip.npz]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from humanoid_mimic_rl import flip_dynamics, flip_opt, g1_backflip, g1_model, reference  # noqa: E402

OPT_RATE_HZ = 100.0  # constraint sampling during optimisation
CLIP_RATE_HZ = 400.0  # exported clip (finite differences in the checker scale with 1/rate^2)
TORQUE_MARGIN = 0.85  # default target: torque <= 85 % of min(force limit, PD authority)
ROOT_FORCE_TOL_FRAC = 0.02  # in-flight root force tolerance, fraction of body weight
ROOT_TORQUE_TOL_NM = 2.0


@dataclasses.dataclass(frozen=True)
class AssetSource:
    """Picklable description of where the G1 assets come from (process-pool factory)."""

    xml_dir: Path | None
    mesh_dir: Path | None

    def assets(self) -> dict[str, bytes]:
        if self.xml_dir is None:
            from mujoco_playground._src import mjx_env  # pylint: disable=import-outside-toplevel
            from mujoco_playground._src.locomotion.g1 import base as g1_base  # pylint: disable=import-outside-toplevel

            mjx_env.ensure_menagerie_exists()
            return g1_base.get_assets()
        assert self.mesh_dir is not None
        files = [*self.xml_dir.glob("*.xml"), *(self.xml_dir / "assets").glob("*"), *self.mesh_dir.glob("*")]
        return {p.name: p.read_bytes() for p in files if p.is_file()}


@dataclasses.dataclass(frozen=True)
class ProblemFactory:
    source: AssetSource
    rate_hz: float
    torque_scale: float = 1.0

    def __call__(self) -> flip_opt.FlipProblem:
        return g1_backflip.make_problem(self.source.assets(), self.rate_hz, self.torque_scale)


def push_mask(problem: flip_opt.FlipProblem) -> np.ndarray:
    """Parameters of the push-off knots (0.55-1.1 s), the toe-off (heel lift) and the take-off time."""
    push = np.zeros(problem.s.num_params, dtype=bool)
    for t_knot in (0.55, 0.75, 0.9, 1.0, 1.1):
        k = int(np.argmin(np.abs(problem.s.knot_times_s - t_knot)))
        for group in problem.s.groups:
            push[problem.knot_param_index(k, group)] = True
    push[-3:] = True  # heel_max, heel_lead, takeoff
    return push


def optimise(
    factory: ProblemFactory, x0: np.ndarray, workers: int, generations: int, torque_margin: float
) -> np.ndarray:
    """Push-off least squares (take-off momentum targets), then CMA-ES with the torque
    margin tightened 2.0 -> 1.2 -> ``torque_margin`` (derivative-free: the event times
    make the residual discontinuous), then a least-squares polish of all parameters at
    ``torque_margin``. Apex-rise / travel limits and touchdown bounds apply throughout."""
    log = lambda s: print(s, flush=True)  # noqa: E731
    problem = factory()
    x = flip_opt.solve_staged(factory, x0, [("takeoff@2.0", push_mask(problem), 600)], workers, log)
    sigma = 0.05
    for margin in (2.0, 1.2, torque_margin):
        stage = f"full@{margin}"
        x = flip_opt.cmaes(factory, x, stage, sigma, generations, workers, log, popsize=2 * workers)
        sigma *= 0.5
    every = np.ones(problem.s.num_params, dtype=bool)
    return flip_opt.solve_staged(factory, x, [(f"full@{torque_margin}", every, 600)], workers, log)


def effective_caps(fb: flip_dynamics.FloatingBaseModel) -> dict[str, list[float]]:
    """[positive, negative] torque cap (N m) per joint at the home pose used by the checker
    (torque motors: the peak torque; servos would add their PD authority)."""
    pos, neg = fb.actuator_caps(fb.m.keyframe("home").qpos[7:])
    return {name: [float(a), float(b)] for name, a, b in zip(g1_model.G1_JOINT_NAMES, pos, neg)}


def build_and_check(
    source: AssetSource, x: np.ndarray, torque_scale: float, torque_margin: float, out: Path
) -> tuple[bool, dict[str, object]]:
    """Export the clip for parameters ``x`` at ``CLIP_RATE_HZ``, run the independent
    inverse-dynamics check, write ``<out>`` and ``<out>.report.json``."""
    problem = ProblemFactory(source, CLIP_RATE_HZ, torque_scale)()
    rollout = problem.rollout(x)
    violations = problem.violations(rollout)
    label = g1_model.torque_scale_label(torque_scale)
    clip = flip_opt.to_clip(
        problem, rollout, g1_model.G1_JOINT_NAMES, g1_model.END_EFFECTOR_SITES, g1_backflip.LICENCE,
        f"{g1_backflip.SOURCE}; {label}",
    )
    reference.save_clip(out, clip)
    fb = problem.fb
    report = flip_dynamics.check_clip(fb, clip, clip.foot_contact, g1_model.G1_JOINT_NAMES)
    root_force_tol = ROOT_FORCE_TOL_FRAC * report.body_weight_n
    passed = report.passes(torque_margin, g1_model.FOOT_FRICTION, root_force_tol, ROOT_TORQUE_TOL_NM)
    pitch = np.unwrap(2.0 * np.arctan2(clip.qpos[:, 5], clip.qpos[:, 3]))
    summary: dict[str, object] = {
        "clip": str(out),
        "torque_scale": torque_scale,
        "model_label": label,
        "frames": clip.num_frames,
        "fps_hz": clip.fps_hz,
        "duration_s": clip.duration_s,
        "takeoff_s": rollout.takeoff_s,
        "touchdown_s": rollout.landing_s,
        "landed": rollout.landed,
        "flight_s": rollout.landing_s - rollout.takeoff_s,
        "takeoff_com_velocity_mps": rollout.takeoff_com_vel.tolist(),
        "takeoff_angular_momentum_nms": rollout.takeoff_ang_mom.tolist(),
        "touchdown_foot_velocity": dict(
            zip(("vx_mps", "vz_mps", "pitch_rate_radps"), rollout.touchdown_foot_vel.tolist())
        ),
        "root_rotation_rad": float(pitch[-1] - pitch[0]),
        "root_displacement_x_m": float(clip.qpos[-1, 0] - clip.qpos[0, 0]),
        "apex_root_height_m": float(np.max(clip.qpos[:, 2])),
        "apex_root_rise_m": float(np.max(clip.qpos[:, 2]) - problem.s.home_qpos[2]),
        "optimiser_constraint_violations": violations,
        "actuation": "torque motors, peak = torque_scale * min(model force limit, G1 EDU 120 N m); see g1_model",
        "heel_lift_rad": float(x[-3]),
        "heel_lift_lead_s": float(x[-2]),
        "effective_torque_cap_at_home_nm": effective_caps(fb),
        "check": dataclasses.asdict(report),
        "check_thresholds": {
            "torque_ratio_max": torque_margin,
            "friction": g1_model.FOOT_FRICTION,
            "flight_root_force_n": root_force_tol,
            "flight_root_torque_nm": ROOT_TORQUE_TOL_NM,
            "min_normal_force_frac_per_foot": flip_dynamics.MIN_NORMAL_FORCE_FRAC,
            "cop_tolerance_m": flip_dynamics.COP_TOLERANCE_M,
        },
        "passed": passed,
    }
    report_path = g1_model.clip_report_path(out)
    report_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return passed, summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "generated" / "g1_backflip.npz")
    p.add_argument("--optimize", action="store_true", help="re-run the trajectory optimisation")
    p.add_argument("--warm-start", action="store_true", help="optimise from the saved solution, not the initial guess")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--generations", type=int, default=600, help="CMA-ES generations per continuation step")
    p.add_argument(
        "--torque-margin", type=float, default=TORQUE_MARGIN,
        help="max torque / cap for the final optimisation stage AND the pass/fail check (report it if != 0.85)",
    )
    p.add_argument(
        "--torque-scales", type=float, nargs="+", default=None,
        help="with --optimize: try these peak-torque multipliers in order (each warm-started from the previous "
        "best) and stop at the first PASS, e.g. 3 4 6. k != 1 is a MODIFIED robot.",
    )
    p.add_argument("--g1-xml-dir", type=Path, default=None)
    p.add_argument("--mesh-dir", type=Path, default=None)
    args = p.parse_args()
    if (args.g1_xml_dir is None) != (args.mesh_dir is None):
        raise SystemExit("--g1-xml-dir and --mesh-dir must be given together")
    source = AssetSource(args.g1_xml_dir, args.mesh_dir)

    if not args.optimize:
        x, scale = g1_backflip.load_solution(g1_backflip.SOLUTION_PATH)
        passed, summary = build_and_check(source, x, scale, args.torque_margin, args.out)
        print(json.dumps(summary, indent=2, allow_nan=False))
        print(f"{'PASS' if passed else 'FAIL'} ({summary['model_label']}): clip {args.out}")
        return 0 if passed else 1

    scales = args.torque_scales or [1.0]
    x = None
    sweep = []
    for scale in scales:
        factory = ProblemFactory(source, OPT_RATE_HZ, scale)
        problem = factory()
        if x is None:
            x = (g1_backflip.load_solution(g1_backflip.SOLUTION_PATH)[0] if args.warm_start
                 else g1_backflip.initial_params(problem))
        print(f"=== torque scale {scale:g}: {g1_model.torque_scale_label(scale)}", flush=True)
        t0 = time.monotonic()
        x = optimise(factory, x, args.workers, args.generations, args.torque_margin)
        wall = time.monotonic() - t0
        passed, summary = build_and_check(source, x, scale, args.torque_margin, args.out)
        g1_backflip.save_solution(
            g1_backflip.SOLUTION_PATH, x, {"torque_scale": scale, "passed": passed, "optimisation_wall_s": wall}
        )
        check = summary["check"]
        sweep.append({"torque_scale": scale, "passed": passed, "wall_s": wall,
                      "torque_ratio_max": check["torque_ratio_max"], "worst_joint": check["worst_joint"]})
        print(f"=== torque scale {scale:g}: {'PASS' if passed else 'FAIL'} in {wall:.0f}s, "
              f"torque_ratio_max {check['torque_ratio_max']:.2f} ({check['worst_joint']})", flush=True)
        if passed:
            break
    sweep_path = args.out.with_suffix(".sweep.json")
    sweep_path.write_text(json.dumps(sweep, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    final = sweep[-1]
    print(f"{'PASS' if final['passed'] else 'FAIL'} at torque scale {final['torque_scale']:g} "
          f"({summary['model_label']}): clip {args.out}, sweep {sweep_path}")
    return 0 if final["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
