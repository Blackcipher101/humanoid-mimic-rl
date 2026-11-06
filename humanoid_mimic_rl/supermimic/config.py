"""SuperMimic hyper-parameters (pure Python, no JAX)."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class SuperMimicConfig:
    """Hyper-parameters (paper values from Table A.5 unless marked "here")."""

    num_iterations: int
    eval_every: int
    num_envs: int = 4096
    buffer_steps_per_env: int = 500  # here (paper 1000) to fit 16 GB with the full trainer
    env_steps_per_iteration: int = 2  # update-to-data 0.5 (paper 0.4)
    warmup_iterations: int = 20  # collection-only iterations before updates (here)
    wm_hidden: tuple[int, ...] = (1024, 1024, 1024)  # here (paper 2048 x 3, 129k motions)
    wm_window: int = 8
    wm_batches_per_iteration: int = 10
    wm_batch_size: int = 2048
    horizon: int = 32
    policy_batch_size: int = 4096
    policy_hidden: tuple[int, ...] = (512, 256, 128)  # here (paper 2048-1024-512-256 decoder)
    critic_hidden: tuple[int, ...] = (256, 128)
    learning_rate: float = 2e-4
    lr_decay_fraction: float = 250_000 / 320_000  # cosine decay over this fraction of iterations
    lr_final_ratio: float = 0.03
    grad_clip_global_norm: float = 1.0
    policy_grad_clip_elementwise: float = 0.1
    discount: float = 0.99
    gae_lambda: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    action_noise_std: float = 0.1
    num_eval_envs: int = 128
    remat_imagination: bool = True  # recompute activations in the backward pass to save memory

    def validate(self) -> None:
        if self.buffer_steps_per_env < self.wm_window + 1:
            raise ValueError("buffer too short for one world-model window")
        if self.warmup_iterations * self.env_steps_per_iteration < self.wm_window + 1:
            raise ValueError("warm-up does not collect enough steps for one window")
        if (self.policy_batch_size * self.horizon) % self.critic_minibatches != 0:
            raise ValueError("critic minibatches must divide policy_batch_size * horizon")
        if self.eval_every < 1 or self.num_iterations < self.eval_every:
            raise ValueError("need 1 <= eval_every <= num_iterations")


def smoke_config() -> SuperMimicConfig:
    return SuperMimicConfig(
        num_iterations=200,
        eval_every=100,
        num_envs=256,
        buffer_steps_per_env=64,
        wm_hidden=(256, 256),
        wm_batches_per_iteration=2,
        wm_batch_size=256,
        horizon=16,
        policy_batch_size=256,
        num_eval_envs=16,
    )


def full_config() -> SuperMimicConfig:
    # Paper: ~0.11 s/iteration on H100s with 2048-wide nets. 15k iterations is
    # a guess for ~1 h on an RTX 5080; the smoke run prints the measured it/s
    # so ``--iterations`` can be rescaled.
    return SuperMimicConfig(num_iterations=15_000, eval_every=1_000)
