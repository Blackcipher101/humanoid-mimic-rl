# humanoid-mimic-rl

Humanoid policies trained in MuJoCo (MJX) with Brax PPO, with video demos:

| task       | env                                                    | method |
|------------|--------------------------------------------------------|--------|
| `walk`     | playground `T1JoystickFlatTerrain` (unchanged)         | playground's joystick reward (velocity tracking + gait shaping) |
| `getup`    | `humanoid_mimic_rl.envs.getup.T1Getup` (new)           | hand-shaped reward: upright + height + gated posture/stillness |
| `backflip` | `humanoid_mimic_rl.envs.backflip.T1Backflip` (new)     | hand-shaped keyframe / phase reward, early termination |
| `mimic`    | `humanoid_mimic_rl.envs.mimic.T1Mimic` (new)           | DeepMimic: tracking reward + reference state init + early termination |
| `supermimic` | same `T1Mimic` env                                   | SuperMimic-style: learned world model + backprop through imagined rollouts (`train_supermimic.py`) |

Robot: **Booster T1** (23 actuated joints, ~30 kg) from MuJoCo Playground /
MuJoCo Menagerie. Playground 0.2.0 ships T1 joystick envs but no humanoid
get-up or backflip, so those are added here on top of playground's T1 model
and base class.

**Status: code only. Nothing in this repository has been trained or run on a
GPU yet.** See "What was verified" below.

## Method

### Common
* Policy at 50 Hz (`ctrl_dt = 0.02 s`), physics at 500 Hz (`sim_dt = 0.002 s`),
  position actuators from the playground T1 model; MuJoCo clamps targets to
  joint ranges.
* Actor observes deployable signals (gyro, gravity in body frame, joint
  angles and velocities, last action, task phase) with playground's uniform
  noise; the critic additionally sees privileged state (linear velocity,
  height, actuator forces, foot contacts).
* Brax PPO with playground's tuned T1 hyper-parameters
  (`humanoid_mimic_rl/configs.py`), 8192 parallel envs.
* Per-step reward = `sum_k scale_k * term_k * ctrl_dt`, clipped at 0 (as in
  playground). Reward kernels are pure functions in
  `humanoid_mimic_rl/rewards.py`, unit-tested with numpy and run with
  `jax.numpy` in the envs.
* Full-body collisions: playground's T1 MJX model lets only the feet touch the
  floor. `t1_model.make_full_collision_xml` derives a model where trunk, head,
  arms and legs collide with the floor (no self-collision, like playground).
  Collision cylinders become capsules of identical end-to-end length (cheaper
  MJX contacts); inertias are unchanged.

### Walk
Playground's env and reward, unchanged. Evaluation disables random pushes and
holds a fixed command (default `vx = 0.5 m/s`).

### Get-up (keyframe-free shaped reward)
Reset drops the robot from 0.6 m with uniformly random orientation and joint
angles and lets it settle for 1 s (80 % of episodes), otherwise starts standing.
Actions are offsets from the *current* joint angles (as playground's
`Go1Getup`). Reward: uprightness `((1+up_z)/2)^2`, torso height (saturating at
the home height 0.665 m), home posture and stillness gated on
"upright (`up_z >= 0.9`) and torso >= 90 % of home height", minus action-rate,
torque, joint-limit and joint-speed penalties. 8 s episodes, no early
termination.

### Backflip (hand-shaped keyframe / phase reward)
Keyframes authored for the T1 (`humanoid_mimic_rl/keyframes.py`): stand,
countermovement crouch with arms back, explosive extension with arm swing,
take-off at 0.90 s, tucked rotation through `-2*pi` about the lateral axis,
touch-down at 1.42 s (0.52 s flight), absorb, recover; 2.8 s total. Joints and
root pitch are interpolated with monotone cubic splines (no overshoot past
keyframes or joint limits). The policy outputs residual targets around the
keyframe pose. Rewards switch on by phase: keyframe pose and pitch tracking
(always), vertical take-off velocity (push-off window), backward pitch rate
(push-off + flight), no foot contact (flight), upright landing on the feet
(after touch-down), home posture (settle window). Episodes start standing (no
reference state init) and terminate when the trunk or head centre drops below
0.2 m.

### Mimic (DeepMimic, Peng et al. 2018)
* Reward `0.65 r_pose + 0.1 r_vel + 0.15 r_ee + 0.1 r_root` with DeepMimic's
  gains (2, 0.1, 40, 10). Root position replaces the CoM term because the
  procedurally generated clip has no measured CoM.
* Reference state initialisation: uniform start time over the clip (10 % at t=0).
* Early termination: trunk/head centre < 0.2 m, root > 0.6 m or > 1.2 rad from
  the reference.
* Deviations: residual joint targets around the reference pose; the final frame
  is held for 0.6 s so landing stabilisation is learned.
* The clip time is `data.time`, so Brax's cached auto-reset (which restores
  `data` but not `info`) restores the phase correctly.

### SuperMimic (model-based, `train_supermimic.py`)
Reimplementation from the paper text of SuperMimic (Liao et al., "SuperMimic:
Scaling Humanoid Whole-Body Control with One Differentiable World Model",
CoRL 2026, arXiv:2610.28480). No code was released with the paper at the time
of writing; nothing here is copied from third-party code. Per iteration
(Algorithm A.1):

1. **Collect**: deterministic policy + `N(0, 0.1^2)` action noise in the MJX
   env (4096 envs), transitions into a per-env ring buffer.
2. **Fit the world model** `s' = f(s, a)` (MLP, LayerNorm + SiLU) on 8-step
   autoregressive windows seeded at logged states (Eq. 1). The state
   representation is yaw/translation invariant (gravity in body frame, body-frame
   velocities); outputs are body-frame increments integrated on SE(3)
   (`supermimic/state.py`, round-trip unit-tested).
3. **Improve the policy** by backpropagating the discounted return of 32-step
   imagined rollouts through the frozen world model (Eq. 2) with soft survival
   weights, bootstrapped by a critic fit to GAE(lambda = 0.95) targets of the
   imagined rollout (Eq. 3). Adam(0.9, 0.95), gradient clipping global-norm 1.0
   then element-wise 0.1, gamma 0.99, cosine LR decay (Table A.5).

Adaptations (single clip, single 16 GB GPU):
* reward / termination / observation are **analytic heads**: the predicted
  state goes through differentiable MJX kinematics and the exact
  `T1Mimic.reward_and_termination` / observation layout the simulator env uses
  (the paper allows analytic heads; it learns a termination probability, here
  termination is a hard indicator under stop-gradient). `self_check` verifies at
  start-up that the analytic heads match the env on `mjx.forward`-consistent
  data and fails loudly otherwise;
* the goal is the clip phase (single clip) rather than 10 future frames;
  reward is this repo's DeepMimic reward rather than BeyondMimic's;
* no domain randomisation, so no sim-context input; uniform RSI instead of
  return-based reset sampling;
* smaller networks / buffer (WM 1024x3, policy 512-256-128, 500 steps/env).
* The real env's sensors lag the state by one 2 ms physics substep (MuJoCo
  computes sensors before integration); the imagined observation does not.
  The self-check prints the size of this lag; observation noise is larger.

### Reference clip
No licence-compatible humanoid backflip mocap exists for the T1 (see
licences), so `scripts/make_reference.py` builds one from the backflip keyframes:
stance frames are fitted so the lowest foot point is on the floor and the feet
do not slide; in flight the whole-body CoM follows the ballistic parabola
between take-off and touch-down (MuJoCo forward kinematics). Generated locally
with mujoco 3.17 (see below): take-off `v_z` 2.36 m/s, apex CoM 0.93 m, no
floor penetration, trunk/head never below 0.52 m. Known limitation: CoM
velocity is discontinuous at take-off (2.5 m/s) and touch-down (2.7 m/s);
physics and RSI absorb this, as in DeepMimic, but it makes the clip only
kinematically plausible.

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

## Results

Fill with measured numbers only (from `eval/metrics.json`, deterministic
policy, 128 episodes). Leave blank until measured.

| task | env steps | train wall time | env steps/s | episode return | success rate | fall rate | task metric |
|---|---|---|---|---|---|---|---|
| walk (vx 0.5 m/s) | | | | | | | fwd speed abs err (m/s): |
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
