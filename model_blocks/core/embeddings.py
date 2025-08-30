from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


class TokenAndPositionEmbedding(nnx.Module):

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        embed_dim: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.token_emb = nnx.Embed(
            num_embeddings=vocab_size,
            features=embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.pos_emb = nnx.Embed(
            num_embeddings=seq_len,
            features=embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        position_embedding = self.pos_emb(positions)
        token_embedding = self.token_emb(x)
        return token_embedding + position_embedding
