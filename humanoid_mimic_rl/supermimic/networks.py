"""Flax networks for SuperMimic: world model, deterministic policy, critic."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jp


class MLP(nn.Module):
    """SiLU MLP; optional LayerNorm after each hidden layer (used by the world model)."""

    hidden: Sequence[int]
    out_dim: int
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for width in self.hidden:
            x = nn.Dense(width)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.variance_scaling(0.01, "fan_in", "uniform"))(x)


class Policy(nn.Module):
    """Deterministic policy ``a = tanh(MLP(o_normalized))`` in [-1, 1]^n."""

    hidden: Sequence[int]
    action_dim: int

    @nn.compact
    def __call__(self, obs_normalized: jax.Array) -> jax.Array:
        return jp.tanh(MLP(self.hidden, self.action_dim)(obs_normalized))


class Critic(nn.Module):
    hidden: Sequence[int]

    @nn.compact
    def __call__(self, x_normalized: jax.Array) -> jax.Array:
        return MLP(self.hidden, 1)(x_normalized)[..., 0]
