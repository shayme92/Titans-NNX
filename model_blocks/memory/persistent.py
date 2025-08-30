from __future__ import annotations
import jax, jax.numpy as jnp
import flax.nnx as nnx


class PersistentTokens(nnx.Module):
    def __init__(self, n_tokens: int, d_model: int, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        key = rngs()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.P = nnx.Param(
            jax.random.normal(key, (n_tokens, d_model), dtype=jnp.float32)
        )

    def tokens(self) -> nnx.Param:
        return self.P
