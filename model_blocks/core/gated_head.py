from __future__ import annotations

from flax import nnx
import jax.numpy as jnp
import jax


class GateNorm(nnx.Module):
    """GLU-style Gate + LayerNorm fusion"""

    def __init__(self, d_model: int, rngs: nnx.Rngs | None = None):
        self.rngs = rngs or nnx.Rngs(0)
        self.norm = nnx.LayerNorm(2 * d_model, rngs=self.rngs)
        self.proj_h = nnx.Linear(2 * d_model, d_model, rngs=self.rngs)
        self.proj_g = nnx.Linear(2 * d_model, d_model, rngs=self.rngs)

    def _fuse(self, y: jnp.ndarray, r: jnp.ndarray):
        u = jnp.concatenate([y, r], axis=-1)
        u = self.norm(u)
        h = self.proj_h(u)
        g = jax.nn.sigmoid(self.proj_g(u))
        o = h * g
        return o

    def __call__(self, y: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
        return self._fuse(y, r)
