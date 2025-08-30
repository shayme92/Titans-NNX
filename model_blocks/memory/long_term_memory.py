import jax
from typing import Any, Tuple, Mapping, List
import jax.numpy as jnp
from flax import nnx
from flax.nnx import State
from flax.nnx.graph import Key
from jax import tree_util as jtu


class LongTermMemory(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        num_memory_tokens: int,
        num_memory_layers: int,
        learning_rate: float = 0.9,
        rngs: nnx.Rngs | None = None,
    ):
        self.rngs = rngs or nnx.Rngs(0)

        self.input_dim = input_dim
        self.num_memory_tokens = num_memory_tokens
        self.num_memory_layers = num_memory_layers
        self.learning_rate = learning_rate
        self.momentum = 0.9
        self.forget_gate = 0.1

        layers = self._create_mlp_layers()
        self.graphdef = nnx.graphdef(nnx.Sequential(*layers))

        self.W_K = nnx.Linear(input_dim, input_dim, rngs=self.rngs)
        self.W_V = nnx.Linear(input_dim, input_dim, rngs=self.rngs)
        self.W_Q = nnx.Linear(input_dim, input_dim, rngs=self.rngs)

    def get_initial_params(self) -> Tuple[State, Mapping["Key", Any]]:
        params = self._get_fresh_params()
        return params, jtu.tree_map(lambda p: jax.lax.full_like(p, 0), params)

    def _create_mlp_layers(self) -> List[nnx.Linear]:
        layers = []
        for layer_idx in range(self.num_memory_layers - 1):
            if layer_idx == 0:
                layers.append(
                    nnx.Linear(self.input_dim, self.num_memory_tokens, rngs=self.rngs)
                )
            else:
                layers.append(
                    nnx.Linear(
                        self.num_memory_tokens, self.num_memory_tokens, rngs=self.rngs
                    )
                )
            layers.append(nnx.silu)

        layers.append(
            nnx.Linear(self.num_memory_tokens, self.input_dim, rngs=self.rngs)
        )

        return layers

    def _get_fresh_params(self) -> State:
        layers = self._create_mlp_layers()
        return nnx.state(nnx.Sequential(*layers), nnx.Param)

    def update_memory(
        self,
        x: jnp.ndarray,
        params: Any,
        prev_surprise: Any,
    ) -> Tuple[Mapping["Key", Any], Any]:
        """
        Updates memory-model params using a weighted sum of per-token grads.
        Returns (new_params, new_prev_surprise)
        """

        n_tokens_in_chunk = x.shape[0]

        def loss_fn(p, xi):
            memory_model = nnx.merge(self.graphdef, p)  # pure, safe inside jit
            k_t = self.W_K(xi)
            v_t = self.W_V(xi)
            pred_v = memory_model(k_t)
            return jnp.mean(jnp.sum((pred_v - v_t) ** 2, axis=-1))

        per_token_grads = jax.vmap(jax.grad(loss_fn, argnums=0), in_axes=(None, 0))(
            params, x
        )

        beta_i = (1.0 - self.forget_gate) ** jnp.arange(1, n_tokens_in_chunk + 1)
        beta_b = beta_i[-1]
        beta_ratios = beta_b / (beta_i + 1e-8)

        cur_surprise = jtu.tree_map(
            lambda g: -self.learning_rate * jnp.tensordot(beta_ratios, g, axes=(0, 0)),
            per_token_grads,
        )

        s_t = jtu.tree_map(
            lambda prev, cur: self.momentum * prev + cur, prev_surprise, cur_surprise
        )

        forget_factor = 1.0 - self.forget_gate
        new_params = jtu.tree_map(lambda p, s: p * forget_factor + s, params, s_t)

        return new_params, s_t

    def __call__(self, x, params: Mapping[Key, Any]) -> jnp.ndarray:
        return self.retrieve(x, params=params)

    def retrieve(self, x: jnp.ndarray, params: Mapping[Key, Any]) -> jnp.ndarray:
        q = self.W_Q(x)
        memory_model = nnx.merge(self.graphdef, params)
        return memory_model(q)
