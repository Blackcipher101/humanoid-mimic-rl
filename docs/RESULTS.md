# Status, results and verification (details)

[Back to README](../README.md)

![Booster T1 walking forward with the trained PPO policy](media/walk_full.gif)

*Trained walk policy, 0.5 m/s forward command ([mp4](media/walk_full.mp4)).*

**Status:** `walk` is trained and evaluated: 128 episodes of 20 s, 0 falls, 100 % success, 0.47 m/s against a 0.5 m/s
command. `getup`, `backflip`, `mimic` and `supermimic` run end to end in smoke tests (1 M steps; no skill learned at
that budget), and their full runs are in progress. See "Results" and "What was verified".

## Results

Measured on workstation1 (RTX 5080, jax 0.9.2, brax 0.14.2, playground 0.2.0), from `eval/metrics.json` with the
deterministic policy over 128 episodes of 20 s. Rows stay blank until measured.

| task | env steps | train wall time | env steps/s | episode return | success rate | fall rate | task metric |
|---|---|---|---|---|---|---|---|
| walk (vx 0.5 m/s) | 151.4 M | 71 min (4255 s incl. JIT) | ~36.9 k | 46.56 ± 0.25 | 1.00 | 0.00 | fwd speed abs err (m/s): 0.052 (mean speed 0.466) |
| getup | | | | | | | median time to stand (s): |
| backflip | | | | | | | max back rotation (rad): / landing success: |
| mimic (keyframed backflip), PPO | | | | | | | max back rotation (rad): / landing success: |
| mimic (keyframed backflip), SuperMimic | (iterations:) | | (it/s:) | | | | max back rotation (rad): / landing success: |

## What was verified (and what was not)

Verified locally (laptop, no JAX): 86 unit tests pass for the reward kernels,
quaternion math, clip I/O / velocities / sampling, keyframe interpolation,
clip fitting (toy kinematics), metrics, and the SuperMimic state
representation (integrate/target round trip, yaw invariance, replay window masks); the full-collision T1 model
compiles with mujoco 3.17 from playground 0.2.0's XMLs + menagerie meshes,
joint order / home pose / ranges match the constants, the home pose stands on
the floor, and the backflip clip was generated without floor penetration.
`pyflakes` is clean.

Not verified: anything that imports JAX/MJX/playground/Flax/Optax (`envs/*`,
`tasks.py`, `train.py`, `train_supermimic.py`, `supermimic/trainer.py`,
`supermimic/networks.py`, `eval_render.py`) has never been executed; APIs were
checked against the playground 0.2.0, brax 0.14.2 and mujoco-mjx 3.17 sources
(Flax/Optax calls from memory: `flax.linen`, `flax.serialization`,
`optax.chain/clip_by_global_norm/clip/adam/cosine_decay_schedule`).
No policy has been trained; whether the T1's actuators (knee 60 N m, hip pitch
45 N m, joint damping 3 N m s/rad) can produce a 0.5 s-flight backflip is
unknown.
