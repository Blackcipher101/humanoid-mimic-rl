# Licences, how to run and budgets (details)

[Back to README](../README.md)

## Licences of assets and data

| item | source | licence | status |
|---|---|---|---|
| Booster T1 MJCF + meshes | mujoco_menagerie `booster_t1` @ `1b86ece5` (pinned by playground 0.2.0), derived from Booster Robotics' public URDF | Apache-2.0 | OK |
| T1 MJX XMLs, envs, base classes | `mujoco_playground` 0.2.0 | Apache-2.0 (rock texture CC0, unused here) | OK |
| Brax | `brax` 0.14.2 | Apache-2.0 | OK |
| Backflip keyframes / generated clip | authored in this repo | same as this repo | OK |
| SuperMimic method | paper arXiv:2610.28480 (reimplemented from the text) | no code or data reused | OK |
| SuperMimic released weights / BONES-SEED motion data | realmimic (CC BY-NC 4.0 weights), BONES licence | non-commercial / restricted | not used |
| LAFAN1, AMASS, SFU-derived DeepMimic clips | not used | non-commercial (LAFAN1 CC BY-NC-ND 4.0, AMASS NC licence) | rejected |

This repository has no licence file yet; the owner must choose one.

## How to run

GPU machine (CUDA 12, headless): venv with `jax[cuda12] mujoco mujoco-mjx
playground brax mediapy imageio imageio-ffmpeg` (optionally `pytest`). The
first run git-clones the full mujoco_menagerie repository into the playground package directory (needs network and disk).

```bash
scripts/run_ws1.sh smoke                 # every task incl. supermimic: tests, clip, short runs, eval, video
scripts/run_ws1.sh full walk             # one task, full budget
IMPL=warp scripts/run_ws1.sh smoke getup # MJX-Warp backend instead of JAX
EXTRA="--num-timesteps 60000000" scripts/run_ws1.sh full backflip
```

Single steps:

```bash
python scripts/make_reference.py                        # -> data/generated/t1_backflip_keyframed.npz
python eval_render.py --reference-only data/generated/t1_backflip_keyframed.npz
python train.py --task getup --profile full [--impl warp] [--num-timesteps N]
python eval_render.py --run-dir runs/getup-full-<stamp> --num-episodes 128 --video-episodes 2
python train_supermimic.py --profile full [--iterations N]   # world-model BPTT on the mimic task
python eval_render.py --run-dir runs/mimic-supermimic-full-<stamp> --playback-speed 0.5
```

`train.py` prints one line per evaluation (step, reward, episode length,
env steps/s, ETA, per-term episode metrics) and writes `progress.jsonl`,
`run_config.json`, `checkpoints/` and `summary.json`. `eval_render.py` writes
`eval/metrics.json` and `eval/rollout_<i>.mp4`.

Laptop / CPU without JAX: `pytest tests` (numpy only; the real-model test runs
when `T1_XML_DIR` and `T1_MESH_DIR` point at playground's `t1/xmls` and
menagerie's `booster_t1/assets` and `mujoco` is installed).

### Budgets
Full: walk 150 M, getup 100 M, backflip 150 M, mimic 150 M env steps;
supermimic 15 k iterations (target ~1 h each on an RTX 5080). **These are
estimates**: the smoke runs report measured env steps/s (PPO) and
iterations/s (SuperMimic); rescale with `--num-timesteps` / `--iterations`.
