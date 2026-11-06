# humanoid-mimic-rl

Humanoid policies trained in MuJoCo (MJX) with Brax PPO, with video demos:

| task       | env                                                    | method |
|------------|--------------------------------------------------------|--------|
| `walk`     | playground `T1JoystickFlatTerrain` (unchanged)         | playground's joystick reward (velocity tracking + gait shaping) |
| `getup`    | `humanoid_mimic_rl.envs.getup.T1Getup` (new)           | hand-shaped reward: upright + height + gated posture/stillness |
| `backflip` | `humanoid_mimic_rl.envs.backflip.T1Backflip` (new)     | hand-shaped keyframe / phase reward, early termination |
| `mimic`    | `humanoid_mimic_rl.envs.mimic.T1Mimic` (new)           | DeepMimic: tracking reward + reference state init + early termination |
| `supermimic` | same `T1Mimic` env                                   | SuperMimic-style: learned world model + backprop through imagined rollouts (`train_supermimic.py`) |
| `g1_backflip` | `humanoid_mimic_rl.envs.g1_flip.G1Backflip` (new)   | Unitree G1, hand-shaped phase reward, prior = optimised feasible clip |
| `g1_mimic`  | `humanoid_mimic_rl.envs.g1_flip.G1Mimic` (new)         | Unitree G1, DeepMimic tracking of the optimised clip |
| `g1_supermimic` | same `G1Mimic` env                                 | SuperMimic-style trainer on G1 (`train_supermimic.py --robot g1`) |

Robot: **Booster T1** (23 actuated joints, ~30 kg) from MuJoCo Playground /
MuJoCo Menagerie. Playground 0.2.0 ships T1 joystick envs but no humanoid
get-up or backflip, so those are added here on top of playground's T1 model
and base class.

![Booster T1 walking forward with the trained PPO policy](docs/media/walk_full.gif)

*Trained walk policy, 0.5 m/s forward command ([mp4](docs/media/walk_full.mp4)).*

**Status:** `walk` is trained and evaluated: 128 episodes of 20 s, 0 falls, 100 % success, 0.47 m/s against a 0.5 m/s
command. `getup`, `backflip`, `mimic` and `supermimic` run end to end in smoke tests (1 M steps; no skill learned at
that budget), and their full runs are in progress. See "Results" and "What was verified".

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

### T1 walk clip from the trained walk policy (`scripts/make_walk_clip.py`)
This is the fast path to a feasible reference for `mimic` / SuperMimic. The
trained T1 walk policy is rolled out in playground's walk env (pushes off,
fixed 0.5 m/s command). The most upright episode that never terminated is
kept, and an 8 s steady window after 3 s of warm-up is stored.

* The clip holds the simulated `qpos`/`qvel` at 50 Hz, so it is physically
  feasible by construction.
* It is rotated to start at the origin heading +x.
* End-effector sites and foot-contact flags come from MuJoCo FK on the
  full-collision T1 used by `T1Mimic`, which has the same bodies and inertias.
* The script prints an approximate inverse-dynamics sanity check
  (`<clip>.report.json`).
* Train on it with `--clip data/generated/t1_walk_from_policy.npz
  --hold-final-s 0`. The 0 hold avoids a reference that stops dead after the clip.

### G1 backflip: dynamically consistent reference (`scripts/make_g1_backflip.py`)
The T1 flip failed because its keyframed clip was not physically possible.
Inverse dynamics of that clip needed up to 3.7 kN of root force in flight and
joint torques up to 11x the T1 limits. The flip tasks therefore moved to the
**Unitree G1** (playground's 29-DoF model, 33.3 kg) with a clip that is
dynamically consistent by construction (`humanoid_mimic_rl/flip_opt.py`):

* decision variables: knots of C2 cubic splines for five joint groups (hip
  pitch, knee, ankle pitch, shoulder pitch left/right-symmetric, plus waist
  pitch; everything else at the home pose) and the take-off time. Waist roll
  stays 0 because the motion is symmetric;
* stance: feet flat and fixed; the root pose follows from closed-chain
  kinematics;
* flight: the root is **integrated from the take-off state with zero external
  wrench** under the prescribed joint motion (RK4, 2.5 ms). The CoM is
  therefore ballistic and the angular momentum about the CoM is conserved by
  construction;
* touch-down: the event at which the foot returns to floor height. It must be
  flat at a pitch of exactly -2*pi with bounded foot speed (small impact,
  reported); landing stance is anchored there;
* constraints from inverse dynamics (`flip_dynamics.py`, all constraints
  disabled so `qfrc_inverse` is the force actuators plus feet must supply):
  * joint torque <= 85 % of the actuator cap;
  * per-foot normal force >= 2.5 % of body weight;
  * friction cone at 80 % of mu = 0.6;
  * centre of pressure inside 80 % of the sole;
  * flight clearance of feet and body; joint limits;
* actuator cap: the peak torque of the flip tasks' torque motors (see
  "Flip-task actuation" below). For position servos the checker would instead
  use min(force limit, PD authority `kp * |ctrl limit - q|`);
* solver: staged bounded least squares, then CMA-ES (derivative-free; the event
  times make the residual discontinuous), with a continuation on the torque
  margin.

`flip_dynamics.check_clip` re-checks the exported clip independently, using
only finite differences of the clip. It reports the in-flight root wrench,
stance normal force / friction / CoP, and the torque/cap ratio per joint. The
numbers are in `data/generated/g1_backflip.report.json`. `make_g1_backflip.py`
exits non-zero when a check fails, so `run_ws1.sh` never trains G1 tasks on an
infeasible clip.

**Status: no feasible G1 backflip found yet.** Laptop runs, mujoco 3.17, 23 workers:

* First attempt (ankle kp 20, waist held at 0): about 25 min of staged least
  squares plus 400 CMA-ES generations at torque margin 3.0. It still failed: no
  touchdown within 0.9 s of flight, waist pitch at 10x its cap around take-off,
  and the CoP 2.6x outside the allowed sole region.
* A/B with equal budgets (take-off least squares, then 120 CMA-ES generations
  at margin 3.0, from the same initial guess):

  | | old: ankle kp 20/2, waist fixed | new: ankle kp 40, waist free |
  |---|---|---|
  | worst torque/cap at start | 69 (ankle pitch) | 35 (ankle pitch) |
  | cost at margin 3.0, after CMA-ES | 2647 | 2437 (-8 %) |
  | cost at margin 0.85, after CMA-ES | 4484 | 4478 |
  | landed within 0.9 s | no | no |
  | worst torque/cap after CMA-ES | 11.3 (waist pitch) | 11.5 (waist pitch) |

The stiffer ankles remove the ankle as the worst joint, but the
residual drops only ~8 % on this budget. The blocker is now the **waist
pitch**: the solutions found whip the pelvis (pitch acceleration -134 rad/s^2
at take-off, up to -516 rad/s^2 in flight). The waist would then need
380-570 N m to carry the ~15 kg upper body, against a cap of about 39 N m.
A feasible flip has to rotate the upper body smoothly (pelvis pitch
acceleration of roughly 35 rad/s^2 or less). Whether the long CMA-ES search on
ws1 finds such a motion is open; the clip is exported only if the
inverse-dynamics check passes.

**Flip-task actuation (owner decision): torque motors at the G1 EDU peak torques.**
The flip tasks `g1_backflip`, `g1_mimic` and `g1_supermimic` replace
playground's soft position servos with torque motors. In those servos the
reachable torque, kp x distance to the control limit, is far below the motor
limits (ankle kp 20 gives ~14 N m against a 50 N m limit). The optimiser and
the checker use the same actuator model.

* Peak torque = min(model force limit, 120 N m). Source: unitree.com/g1, spec
  table, G1 EDU "Maximum Torque of Knee Joint: 120 N.m" (G1: 90 N.m). Its
  footnote says "The maximum torque of the joint motors of the whole machine is
  different. This is the maximum torque of the largest joint motor among them."
* Unitree publishes no other per-joint figure, so every other joint keeps the
  model's force limit (Unitree g1_description effort limits, via playground /
  Menagerie), capped at 120 N m.
* Resulting peaks (N m): knee 120 and hip roll 120 (both were 139); hip
  pitch/yaw 88, ankle pitch/roll 50, waist yaw 88, waist roll/pitch 50,
  shoulder/elbow/wrist roll 25, wrist pitch/yaw 5 (unchanged).
* Checker cap = that peak torque (no PD-authority term), with the 0.85 margin.
* Env actuation (`G1FullCollisionEnv.TORQUE_MODE`): the policy keeps outputting
  joint position targets. A PD-to-torque layer evaluated every 2 ms physics
  substep gives `tau = clip(kp (target - q) - kd qd, +-peak)`, with
  `kp = peak / 0.25 rad` (full torque at 0.25 rad of error), `kd = 0.02 s * kp`,
  and targets allowed 0.25 rad beyond the joint range. The full peak torque is
  therefore reachable everywhere, so the checker cap matches the env.
* Why a PD layer and not raw torque outputs: the residual-on-reference
  position actions keep the same action space and scale as the PPO/SuperMimic
  mimic and backflip tasks, which train well in MJX. Raw torque policies
  usually need far more samples. These kp/kd are a design choice, not
  Unitree's hardware gains, and are untested in training so far.
* Earlier attempts (playground servos, then Unitree-sourced ankle kp 40) are
  superseded. The A/B above showed that the servo caps, not the optimiser, were
  binding.

Local trend check for the torque-motor formulation (laptop, take-off least squares
then 120 CMA-ES generations at margin 2.0, from the initial guess): still
infeasible. No touchdown within 0.9 s, over-rotation (touchdown pitch -7.5 vs
-2*pi rad), and the waist pitch is the worst joint at 10.3x its 50 N m peak
(11.5x with the servo caps at the same budget). Net travel is now within bounds
(-0.25 m) and the apex rise is lower (0.74 m). A ws1 run of the previous
formulation (`be24da8`, 12 workers, ~25 min) reached a 4.2x worst torque ratio
with a full rotation, so the long search is the real test.

Optimiser additions in this round:
* toe-off: the feet may pivot about the sole's front edge before take-off
  (heel lift up to 0.6 rad over the last 0.05-0.25 s; the CoP must then stay
  within 2 cm of the toe edge);
* the floor CoP is computed on the ground plane against the touching sole
  region;
* root apex rise <= 0.5 m above home and net travel <= 0.5 m (a lower, tighter
  flip; the ws1 run reached a 0.83 m rise and 1.13 m travel);
* torque-margin continuation 2.0 -> 1.2 -> 0.85, ending at the target 0.85;
* the clip is exported at 400 Hz, so the checker's finite differences are 4x
  more accurate than at 200 Hz.

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

G1 verification (laptop, mujoco 3.17, no JAX):
* the full-collision G1 compiles; joint order and home pose match; lying on its
  back it contacts pelvis, torso, shins and hands;
* on a synthetic biped, the flip-dynamics code is verified for:
  * static stance wrench and CoP;
  * free-fall root acceleration;
  * PD authority caps;
  * flat, fixed stance feet;
  * ballistic CoM (acceleration -9.81 m/s^2 to 1e-4);
  * conserved angular momentum in flight;
  * `check_clip` reporting near-zero in-flight root wrench for a simulated flight.

The G1 env classes (`envs/g1_full_collision.py`, `envs/g1_flip.py`) and the
G1 SuperMimic path have never run (no JAX locally).

## Licences of assets and data

| item | source | licence | status |
|---|---|---|---|
| Unitree G1 29-DoF MJX XML (`locomotion/g1/xmls`) | `mujoco_playground` 0.2.0 | Apache-2.0 | OK |
| Unitree G1 meshes; pelvis/torso/head collider sizes | mujoco_menagerie `unitree_g1` @ `1b86ece5` (`g1_mjx.xml`), Unitree Robotics | BSD-3-Clause (copyright notice retained) | OK |
| G1 backflip clip | generated by `flip_opt` in this repo | same as this repo | OK |
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
