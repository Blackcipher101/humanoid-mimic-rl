"""Environment construction per task (requires JAX / MJX / mujoco_playground)."""

from __future__ import annotations

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


def mimic_overrides(eval_mode: bool, termination: str | None) -> dict[str, object]:
    """Config overrides shared by the mimic tasks (``mimic`` and ``g1_mimic``)."""
    out: dict[str, object] = {}
    if eval_mode:
        out["rsi_start_prob"] = 1.0  # always start at t = 0
    if termination is not None:
        out["termination"] = termination
    return out


def make_env(
    task: str,
    impl: str,
    num_worlds: int,
    reference_path: Path | None = None,
    eval_mode: bool = False,
    hold_final_s: float | None = None,
    torque_scale: float | None = None,
    termination: str | None = None,
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
    if task == "mimic":
        path = (reference_path or DEFAULT_REFERENCE_PATH).resolve()
        cfg = mimic.default_config()
        clip = reference.load_clip(path)
        overrides["reference_path"] = str(path)
        hold = cfg.hold_final_s if hold_final_s is None else float(hold_final_s)
        overrides["hold_final_s"] = hold
        overrides["episode_length"] = mimic.episode_length_for(clip, cfg.ctrl_dt, hold)
        overrides.update(mimic_overrides(eval_mode, termination))
        return mimic.T1Mimic(cfg, overrides)
    if task in ("g1_backflip", "g1_mimic"):
        path = (reference_path or G1_REFERENCE_PATH).resolve()
        scale = g1_model.resolve_torque_scale(path, torque_scale)
        if task == "g1_backflip":
            return g1_flip.G1Backflip(g1_flip.backflip_config(str(path), scale), overrides)
        cfg = g1_flip.mimic_config(scale)
        clip = reference.load_clip(path)
        hold = cfg.hold_final_s if hold_final_s is None else float(hold_final_s)
        overrides["reference_path"] = str(path)
        overrides["hold_final_s"] = hold
        overrides["episode_length"] = mimic.episode_length_for(clip, cfg.ctrl_dt, hold)
        overrides.update(mimic_overrides(eval_mode, termination))
        return g1_flip.G1Mimic(cfg, overrides)
    raise ValueError(f"unknown task {task!r}")


def episode_length(env: mjx_env.MjxEnv) -> int:
    """Episode length (control steps) from the environment config."""
    n = int(env.unwrapped._config.episode_length)  # pylint: disable=protected-access
    if n <= 0:
        raise ValueError("environment config has no positive episode_length")
    return n
