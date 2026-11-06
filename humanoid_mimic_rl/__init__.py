"""Humanoid motion-imitation RL on MuJoCo Playground (MJX + Brax PPO).

Modules that only depend on numpy (``quat``, ``rewards``, ``reference``,
``keyframes``, ``metrics``, ``configs``, ``t1_model``) are importable without
JAX so they can be unit-tested on machines without a GPU stack. The
environment modules under ``humanoid_mimic_rl.envs`` and ``tasks`` require
JAX, MJX and ``mujoco_playground``.
"""
