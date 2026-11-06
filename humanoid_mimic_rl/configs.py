"""Task registry and PPO budgets (pure Python, no JAX).

Every task has a ``full`` profile (the real run) and a ``smoke`` profile
(minutes; checks that compilation, training, checkpointing and evaluation
work end to end, says nothing about policy quality).

The full step budgets are starting points chosen to fit roughly one hour
per task on a single RTX 5080. They are NOT measured yet: the smoke run
prints environment steps per second, and ``train.py`` prints an ETA after
every evaluation, so the budgets can be scaled with ``--num-timesteps``
once the real throughput is known. PPO hyper-parameters follow
playground's tuned T1 joystick config (``locomotion_params.brax_ppo_config``).
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    policy_obs_key: str = "state"
    value_obs_key: str = "privileged_state"


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """Arguments of ``brax.training.agents.ppo.train.train`` that we set."""

    num_timesteps: int
    num_evals: int
    num_envs: int = 8192
    num_eval_envs: int = 128
    batch_size: int = 256
    num_minibatches: int = 32
    unroll_length: int = 20
    num_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    entropy_cost: float = 5e-3
    discounting: float = 0.97
    clipping_epsilon: float = 0.2
    max_grad_norm: float = 1.0
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    num_resets_per_eval: int = 1
    network: NetworkConfig = NetworkConfig()

    @property
    def env_steps_per_update(self) -> int:
        """Environment steps consumed per PPO training step (action_repeat = 1)."""
        return self.batch_size * self.num_minibatches * self.unroll_length

    def validate(self) -> None:
        """Check the divisibility rules Brax PPO asserts at runtime."""
        if (self.batch_size * self.num_minibatches) % self.num_envs != 0:
            raise ValueError(
                f"batch_size*num_minibatches={self.batch_size * self.num_minibatches} "
                f"must be divisible by num_envs={self.num_envs}"
            )
        if self.num_evals < 1:
            raise ValueError("num_evals must be >= 1")
        if self.num_timesteps < self.env_steps_per_update * self.num_evals:
            raise ValueError(
                f"num_timesteps={self.num_timesteps} is less than one PPO update "
                f"({self.env_steps_per_update} steps) per evaluation"
            )

    def brax_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``ppo.train`` (network handled separately)."""
        out = dataclasses.asdict(self)
        del out["network"]
        return out


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    full: PPOConfig
    smoke: PPOConfig
    domain_randomization: bool = False

    def profile(self, name: str) -> PPOConfig:
        if name == "full":
            return self.full
        if name == "smoke":
            return self.smoke
        raise ValueError(f"unknown profile {name!r}; expected 'full' or 'smoke'")


def _smoke() -> PPOConfig:
    return PPOConfig(
        num_timesteps=1_000_000,
        num_evals=3,
        num_envs=1024,
        num_eval_envs=64,
        batch_size=256,
        num_minibatches=4,
    )


TASKS: dict[str, TaskSpec] = {
    "walk": TaskSpec(
        name="walk",
        description="Joystick velocity tracking, playground T1JoystickFlatTerrain (unchanged env).",
        full=PPOConfig(num_timesteps=150_000_000, num_evals=15),
        smoke=_smoke(),
    ),
    "getup": TaskSpec(
        name="getup",
        description="Get up from a random fall (humanoid_mimic_rl.envs.getup.T1Getup).",
        full=PPOConfig(num_timesteps=100_000_000, num_evals=10),
        smoke=_smoke(),
    ),
    "backflip": TaskSpec(
        name="backflip",
        description="Standing backflip, keyframe/phase reward (envs.backflip.T1Backflip).",
        full=PPOConfig(num_timesteps=150_000_000, num_evals=15),
        smoke=_smoke(),
    ),
    "mimic": TaskSpec(
        name="mimic",
        description="DeepMimic tracking of a reference clip, default: keyframed backflip (envs.mimic.T1Mimic).",
        full=PPOConfig(num_timesteps=150_000_000, num_evals=15),
        smoke=_smoke(),
    ),
}


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name]
