import jax.numpy as jnp
from flax import nnx
import optax
from typing import Tuple


@nnx.jit
def cross_entropy(model, batch: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()
    return loss, logits
