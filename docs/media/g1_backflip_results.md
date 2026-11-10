# G1 backflip videos (simulation)

All videos: **G1 with 5x published peak torque - a modified model, not the real robot's limits** (label burned
into each frame). MuJoCo MJX simulation only; nothing here ran on a physical robot.

Policy: PPO `g1_mimic` (DeepMimic tracking with reference state initialisation) on a G1 backflip clip retargeted
from MimicKit's `humanoid_backflip` motion, then fine-tuned with landing-focused starts (`--rsi-focus 0.9 1.44 0.5`)
and a looser root-height termination (`--max-root-height-err 0.4`). The clip itself is not in the repo (MimicKit's
motion data has no stated licence).

| file | what it shows |
|---|---|
| `g1_backflip_landed_a.mp4` (+ `_slow`, 0.5x) | landed episode, policy after 650M steps |
| `g1_backflip_landed_b.mp4` (+ `_slow`) | landed episode, final policy (+ looser tilt termination) |
| `g1_backflip_landed_c.mp4` | landed episode, final policy |
| `g1_backflip_typical_fail.mp4` | a typical episode: full rotation, lands, then falls back |

The landed videos are **selected** episodes. Success rate of the final policies: about 5-11% of 128 episodes land
(full rotation within 0.5 rad, no termination, upright and at standing height with a foot on the floor at the end);
75-85% complete the full rotation. Most failures land upright and then hop or sink into a squat.
