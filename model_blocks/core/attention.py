from __future__ import annotations
import jax.numpy as jnp
from flax import nnx


def causal_attention_mask(seq_len: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


class MHAttention(nnx.Module):
    """
    A multi-head attention block with causal self-attention and layer normalization.
    """

    def __init__(self, embed_dim: int, num_heads: int, rngs: nnx.Rngs | None = None):

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        rngs = rngs or nnx.Rngs(0)

        self.layer_norm1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=embed_dim, rngs=rngs
        )

    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        seq_len = inputs.shape[0]

        causal_mask = causal_attention_mask(seq_len)

        if mask is not None:
            mask = causal_mask & mask
        else:
            mask = causal_mask

        attention_output = self.mha(
            inputs_q=self.layer_norm1(inputs),
            mask=mask,
            decode=False,
        )

        x = inputs + attention_output

        return x
