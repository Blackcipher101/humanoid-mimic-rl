"""Environment construction per task (requires JAX / MJX / mujoco_playground)."""

from __future__ import annotations

import json
from pathlib import Path

from mujoco_playground import registry
from mujoco_playground._src import mjx_env

from humanoid_mimic_rl import g1_model, reference
from humanoid_mimic_rl.envs import backflip, g1_flip, getup, mimic

REPO_ROOT = Path(__file__).resolve().parent.parent
WALK_ENV_NAME = "T1JoystickFlatTerrain"
DEFAULT_REFERENCE_PATH = REPO_ROOT / "data" / "generated" / "t1_backflip_keyframed.npz"
G1_REFERENCE_PATH = REPO_ROOT / "data" / "generated" / "g1_backflip.npz"

# Upper bound on simultaneous contacts per world for the full-collision T1 / G1
# (13 / 6 were measured lying on the back with the C MuJoCo engine; 32 leaves margin).
# Only used by the MJX "warp" implementation, which preallocates contacts.
CONTACTS_PER_WORLD = 32


def mimic_overrides(
    eval_mode: bool, termination: str | None, max_tilt_err: float | None = None,
    rsi_focus: tuple[float, float, float] | None = None, max_root_height_err_m: float | None = None,
) -> dict[str, object]:
    """Config overrides shared by the mimic tasks (``mimic`` and ``g1_mimic``).

    ``max_tilt_err``: ``height_tilt`` termination threshold on ``|g_z_body - g_z_ref|`` (env default 0.8;
    validated by the env to lie in (0, 2], 2 disables the tilt test).
    ``max_root_height_err_m``: ``height_tilt`` root height threshold (env default 0.25 m).
    ``rsi_focus``: ``(start_s, end_s, prob)`` focused-RSI window, training only (eval always starts at t = 0).
    """
    out: dict[str, object] = {}
    if eval_mode:
        out["rsi_start_prob"] = 1.0  # always start at t = 0
    if termination is not None:
        out["termination"] = termination
    if max_tilt_err is not None:
        out["max_tilt_err"] = float(max_tilt_err)
    if max_root_height_err_m is not None:
        out["max_root_height_err_m"] = float(max_root_height_err_m)
    if rsi_focus is not None and not eval_mode:
        start_s, end_s, prob = (float(v) for v in rsi_focus)
        out.update(rsi_focus_start_s=start_s, rsi_focus_end_s=end_s, rsi_focus_prob=prob)
    return out


ROUTINE_OPTIONS = ("flip_bonus_fraction", "kick_bonus_fraction", "stand_bonus_fraction", "gated")


def routine_options_from_args(args: object) -> dict[str, object]:
    """``routine_options`` for ``make_env`` from the ``--routine-*`` CLI flags (unset flags are omitted)."""
    out = {
        "flip_bonus_fraction": getattr(args, "routine_flip_bonus", None),
        "kick_bonus_fraction": getattr(args, "routine_kick_bonus", None),
        "stand_bonus_fraction": getattr(args, "routine_stand_bonus", None),
        "gated": True if getattr(args, "routine_gated", False) else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def routine_overrides(
    clip_path: Path, routine: bool | None, options: dict[str, object] | None
) -> dict[str, object]:
    """Config overrides enabling the routine reward (``humanoid_mimic_rl.routine``) for ``clip_path``.

    ``routine=None`` enables it iff the clip's report has a ``routine`` section; ``True`` requires one
    (``ValueError`` otherwise); ``False`` keeps the plain tracking reward. ``options`` (keys in
    ``ROUTINE_OPTIONS``: bonus fractions, ``gated``) override the config defaults and need the routine reward
    to be enabled.
    """
    options = options or {}
    unknown = set(options) - set(ROUTINE_OPTIONS)
    if unknown:
        raise ValueError(f"unknown routine options {sorted(unknown)}; known: {ROUTINE_OPTIONS}")
    report = g1_model.clip_report_path(clip_path)
    has_section = report.is_file() and "routine" in json.loads(report.read_text(encoding="utf-8"))
    enabled = has_section if routine is None else routine
    if enabled and not has_section:
        raise ValueError(f"--routine needs a clip report with a 'routine' section ({report}); "
                         "build the clip with scripts/make_g1_routine.py")
    if not enabled:
        if options:
            raise ValueError("routine options given but the routine reward is not enabled")
        return {}
    return {"routine.enabled": True, "routine.report_path": str(report),
            **{f"routine.{k}": v for k, v in options.items()}}


def routine_run_config(env: mjx_env.MjxEnv) -> dict[str, object] | None:
    """``run_config.json`` entry of the routine reward (config plus the resolved terminal bonus), or ``None``."""
    cfg = env.unwrapped._config.get("routine")  # pylint: disable=protected-access
    if cfg is None or not cfg.enabled:
        return None
    return {**cfg.to_dict(), "bonuses": dict(zip(("flip", "kick", "stand"), env.unwrapped.routine_bonuses))}


def make_env(
    task: str,
    impl: str,
    num_worlds: int,
    reference_path: Path | None = None,
    eval_mode: bool = False,
    hold_final_s: float | None = None,
    torque_scale: float | None = None,
    termination: str | None = None,
    allow_infeasible_clip: bool = False,
    routine: bool | None = None,
    routine_options: dict[str, object] | None = None,
    max_tilt_err: float | None = None,
    rsi_focus: tuple[float, float, float] | None = None,
    max_root_height_err_m: float | None = None,
    flight_start_prob: float | None = None,
) -> mjx_env.MjxEnv:
    """Build the environment for ``task``.

    Args:
      task: one of ``configs.TASKS``.
      impl: MJX implementation, ``"jax"`` or ``"warp"``.
      num_worlds: number of parallel environments this instance will be
        vmapped over (sizes the warp contact buffer).
      reference_path: clip for ``mimic`` (default ``DEFAULT_REFERENCE_PATH``) or for
        ``g1_backflip`` / ``g1_mimic`` (default ``G1_REFERENCE_PATH``).
      hold_final_s: mimic tasks only: seconds the reference holds its last frame
        after the clip ends (default: the env config's 0.6 s; use 0 for cyclic
        motions such as walking, where an abrupt stop is not wanted).
      torque_scale: G1 flip tasks only: multiplier on the published peak torques.
        Default and required value: the one recorded in ``<clip>.report.json``.
      termination: mimic tasks only: ``"root_pose"`` or ``"height_tilt"``
        (default: the env config's ``"root_pose"``); see ``envs.mimic``.
      allow_infeasible_clip: G1 mimic/flip tasks only: accept a clip whose report FAILED the
        torque feasibility check (see ``g1_model.resolve_torque_scale``).
      routine, routine_options: ``g1_mimic`` only: multi-stage routine reward, see
        ``routine_overrides`` and ``humanoid_mimic_rl.routine``.
      max_tilt_err: mimic tasks only: ``height_tilt`` tilt threshold (default: the env config's 0.8).
      rsi_focus: mimic tasks only: ``(start_s, end_s, prob)`` focused reference state initialisation.
      max_root_height_err_m: mimic tasks only: ``height_tilt`` height threshold (default: the env config's 0.25 m).
      flight_start_prob: ``g1_backflip`` only: probability of starting an episode in the clip's flight
        (``G1Backflip.reset``); ignored with ``eval_mode`` (evaluation always starts standing).
      eval_mode: for ``walk``, disables the random pushes so velocity tracking
        is measured undisturbed. For ``mimic`` / ``g1_mimic``, every episode
        starts at clip time 0 (``rsi_start_prob=1``) so the whole clip is scored.
        Observation noise is kept in all tasks.
    """
    if impl not in ("jax", "warp"):
        raise ValueError(f"impl must be 'jax' or 'warp', got {impl!r}")
    # The custom envs read T1 meshes from playground's menagerie checkout,
    # which registry.load() would otherwise clone lazily.
    mjx_env.ensure_menagerie_exists()
    if task == "walk":
        overrides: dict[str, object] = {"impl": impl}
        if eval_mode:
            overrides["push_config.enable"] = False
        return registry.load(WALK_ENV_NAME, config_overrides=overrides)

    overrides = {"impl": impl, "naconmax": CONTACTS_PER_WORLD * max(num_worlds, 1)}
    if task == "getup":
        return getup.T1Getup(getup.default_config(), overrides)
    if task == "backflip":
        return backflip.T1Backflip(backflip.default_config(), overrides)
    if (max_tilt_err is not None or rsi_focus is not None or max_root_height_err_m is not None) and task not in (
            "mimic", "g1_mimic"):
        raise ValueError(f"max_tilt_err / rsi_focus / max_root_height_err_m are only available for the mimic tasks, not {task!r}")
    if flight_start_prob is not None and task != "g1_backflip":
        raise ValueError(f"flight_start_prob is only available for g1_backflip, not {task!r}")
    if routine and task != "g1_mimic":
        raise ValueError(f"the routine reward is only available for g1_mimic, not {task!r}")
    if task == "mimic":
        path = (reference_path or DEFAULT_REFERENCE_PATH).resolve()
        cfg = mimic.default_config()
        clip = reference.load_clip(path)
        overrides["reference_path"] = str(path)
        hold = cfg.hold_final_s if hold_final_s is None else float(hold_final_s)
        overrides["hold_final_s"] = hold
        overrides["episode_length"] = mimic.episode_length_for(clip, cfg.ctrl_dt, hold)
        overrides.update(mimic_overrides(eval_mode, termination, max_tilt_err, rsi_focus, max_root_height_err_m))
        return mimic.T1Mimic(cfg, overrides)
    if task in ("g1_backflip", "g1_mimic"):
        path = (reference_path or G1_REFERENCE_PATH).resolve()
        scale = g1_model.resolve_torque_scale(path, torque_scale, allow_infeasible_clip)
        if task == "g1_backflip":
            if flight_start_prob is not None and not eval_mode:
                overrides["flight_start_prob"] = float(flight_start_prob)
            return g1_flip.G1Backflip(g1_flip.backflip_config(str(path), scale), overrides)
        cfg = g1_flip.mimic_config(scale)
        clip = reference.load_clip(path)
        hold = cfg.hold_final_s if hold_final_s is None else float(hold_final_s)
        overrides["reference_path"] = str(path)
        overrides["hold_final_s"] = hold
        overrides["episode_length"] = mimic.episode_length_for(clip, cfg.ctrl_dt, hold)
        overrides.update(mimic_overrides(eval_mode, termination, max_tilt_err, rsi_focus, max_root_height_err_m))
        overrides.update(routine_overrides(path, routine, routine_options))
        return g1_flip.G1Mimic(cfg, overrides)
    raise ValueError(f"unknown task {task!r}")


def episode_length(env: mjx_env.MjxEnv) -> int:
    """Episode length (control steps) from the environment config."""
    n = int(env.unwrapped._config.episode_length)  # pylint: disable=protected-access
    if n <= 0:
        raise ValueError("environment config has no positive episode_length")
    return n
