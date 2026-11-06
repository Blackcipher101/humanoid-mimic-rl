# humanoid-mimic-rl

Booster T1 humanoid policies trained in MuJoCo (MJX) with Brax PPO: walk, get-up, backflip, DeepMimic-style
imitation and a SuperMimic-style world-model reimplementation.

| Get-up (v5): from lying on its back (left) and after slipping into the splits (right) |
|---|
| [![Booster T1 getting up from the floor with the trained PPO policy](docs/media/getup_full.gif)](docs/media/getup_full.mp4) |
| [mp4](docs/media/getup_full.mp4) |

| Trained walk policy, 0.5 m/s forward command |
|---|
| [![Booster T1 walking forward with the trained PPO policy](docs/media/walk_full.gif)](docs/media/walk_full.mp4) |
| [mp4](docs/media/walk_full.mp4) |

## Results

Workstation RTX 5080, deterministic policy, 128 episodes each (`eval/metrics.json`; walk 20 s episodes, get-up 8 s):

| task | env steps | train wall time | episode return | success rate | fall rate | task metric |
|---|---|---|---|---|---|---|
| walk (vx 0.5 m/s) | 151.4 M | 71 min (4255 s incl. JIT) | 46.56 ± 0.25 | 1.00 | 0.00 | fwd speed abs err 0.052 m/s (mean speed 0.466) |
| get-up (v5) | 100.3 M | 82 min (4920 s incl. JIT) | 42.57 ± 5.81 | 0.99 (127/128) | 0.00 | median time to stand 1.32 s (mixed starts) |

Get-up success = standing for the final 1 s of the 8 s episode. The 128 episodes mix starts: dropped from 0.6 m in a
random orientation, snapshots of a pushed fall, and standing still. Floor-only success is not measured separately yet.

| task | env | method | status |
|---|---|---|---|
| `walk` | playground `T1JoystickFlatTerrain` (unchanged) | playground's joystick reward | trained and evaluated |
| `getup` | `T1Getup` (new) | home-relative actions, standing-gated posture reward, fall-snapshot start curriculum | **trained**: v5 stands in 127/128 (v1 sat, v2 curled up, v3/v4 0 rises) |
| `backflip` | `T1Backflip` (new) | hand-shaped keyframe / phase reward | failed on T1 (clip infeasible, torques ~10x limits); moving to Unitree G1 |
| `mimic` | `T1Mimic` (new) | DeepMimic tracking reward + RSI + early termination | failed on T1 (same infeasible clip); moving to Unitree G1 |
| `supermimic` | same `T1Mimic` env | learned world model + backprop through imagined rollouts | smoke test only; full run after the G1 clip |

Full runs of `backflip`, `mimic` and `supermimic` are in progress.

## Quick start

```bash
pytest tests                              # laptop / CPU, numpy only
scripts/run_ws1.sh smoke                  # GPU: every task, tests, clip, short runs, eval, video
scripts/run_ws1.sh full walk              # one task, full budget
python eval_render.py --run-dir runs/<task>-full-<stamp> --num-episodes 128 --video-episodes 2
```

## More

- [docs/METHOD.md](docs/METHOD.md): the task table, robot, common setup (rates, observations, PPO, full-body
  collisions) and each task's reward: walk, get-up, backflip, DeepMimic, SuperMimic adaptations, the reference clip.
- [docs/USAGE.md](docs/USAGE.md): asset and data licences, environment, single training / eval steps, budgets.
- [docs/RESULTS.md](docs/RESULTS.md): status, the full results table (rows blank until measured) and what was and
  was not verified.

Known limitations: the backflip reference clip is generated from keyframes (only kinematically plausible), and whether
the T1's actuators can produce a 0.5 s-flight backflip is unknown.
Assets are Apache-2.0 (Booster T1 via MuJoCo Menagerie / Playground); this repository has no licence file yet.
