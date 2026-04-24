"""Model definition for Craftax PPO.

This file intentionally only exposes the actor-critic model.
"""

from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
from jax import nn as jnn


class ActorCritic(nn.Module):
    """Simple MLP actor-critic for symbolic Craftax observations.

    Returns:
        logits: [batch, num_actions]
        value:  [batch]
    """

    num_actions: int
    hidden_size: int = 512
    num_hidden_layers: int = 3
    activation: str = "tanh"
    orthogonal_init: bool = True
    actor_output_scale: float = 0.01
    critic_output_scale: float = 1.0

    def _activation(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        if self.activation == "tanh":
            return jnp.tanh
        if self.activation == "relu":
            return jnn.relu
        if self.activation == "silu":
            return jnn.silu
        raise ValueError(f"Unknown activation: {self.activation}")

    def _kernel_init(self, scale: float):
        if self.orthogonal_init:
            return nn.initializers.orthogonal(scale)
        return nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = obs.astype(jnp.float32)
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))

        act = self._activation()

        actor = x
        critic = x
        for i in range(self.num_hidden_layers):
            actor = nn.Dense(
                self.hidden_size,
                kernel_init=self._kernel_init(jnp.sqrt(2.0)),
                bias_init=nn.initializers.constant(0.0),
                name=f"actor_fc{i}",
            )(actor)
            actor = act(actor)

            critic = nn.Dense(
                self.hidden_size,
                kernel_init=self._kernel_init(jnp.sqrt(2.0)),
                bias_init=nn.initializers.constant(0.0),
                name=f"critic_fc{i}",
            )(critic)
            critic = act(critic)

        logits = nn.Dense(
            self.num_actions,
            kernel_init=self._kernel_init(self.actor_output_scale),
            bias_init=nn.initializers.constant(0.0),
            name="actor_logits",
        )(actor)
        value = nn.Dense(
            1,
            kernel_init=self._kernel_init(self.critic_output_scale),
            bias_init=nn.initializers.constant(0.0),
            name="critic_value",
        )(critic)
        return logits, jnp.squeeze(value, axis=-1)


__all__ = ["ActorCritic"]
