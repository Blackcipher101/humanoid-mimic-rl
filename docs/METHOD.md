# Tasks and method (details)

[Back to README](../README.md)

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
