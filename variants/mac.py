from __future__ import annotations

from typing import Tuple
import jax
import jax.numpy as jnp
from flax import nnx

from model_blocks.core.attention import MHAttention
from model_blocks.memory.long_term_memory import LongTermMemory
from model_blocks.memory.persistent import PersistentTokens
from model_blocks.core.embeddings import TokenAndPositionEmbedding
from model_blocks.core.gated_head import GateNorm


class TitanMAC(nnx.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        seq_len: int,
        chunk_size: int,
        n_persist: int,
        n_mem_tokens: int,
        mem_hidden: int,
        pad_id: int = -float("inf"),
        vocab_size: int = 256,
        rngs: nnx.Rngs | None = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_persist = n_persist
        self.n_mem_tokens = n_mem_tokens
        self.pad_id = pad_id
        self.chunk_size = chunk_size
        self.embedding = TokenAndPositionEmbedding(
            seq_len, vocab_size, d_model, rngs=rngs
        )
        self.pmem = PersistentTokens(n_persist, d_model, rngs=rngs)
        self.ltm = LongTermMemory(d_model, n_mem_tokens, mem_hidden, rngs=rngs)
        self.attention_block = MHAttention(d_model, n_heads, rngs=rngs)
        self.gate_norm = GateNorm(d_model, rngs=rngs)

    def _project_to_vocab_dim(self, ot: jnp.ndarray):
        return self.embedding.token_emb.attend(ot)

    def _get_num_of_chunks_and_pad_len(self, seq_len: int) -> Tuple[int, int]:
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        padded_len = num_chunks * self.chunk_size
        pad_len = padded_len - seq_len
        return num_chunks, pad_len

    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len = tokens.shape
        num_chunks, pad_len = self._get_num_of_chunks_and_pad_len(seq_len)

        pad = jnp.full((batch_size, pad_len), self.pad_id, dtype=tokens.dtype)
        tokens_padded = jnp.concatenate([tokens, pad], axis=1)

        mask_padded = tokens_padded != self.pad_id

        x = self.embedding(tokens_padded)

        x_chunks = x.reshape(batch_size, num_chunks, self.chunk_size, -1)
        mask_chunks = mask_padded.reshape(batch_size, num_chunks, self.chunk_size)

        def forward_update_long_term_memory(carry, xs_chunk):
            prev_long_term_memory, prev_surprise = carry

            chunk_x, chunk_mask = xs_chunk

            ht = self.ltm.retrieve(chunk_x, prev_long_term_memory)
            p = self.pmem.P
            chunk_x_with_mem = jnp.concatenate([p, ht, chunk_x], axis=0)

            num_p = self.pmem.n_tokens
            tilde_mask = jnp.concatenate(
                [jnp.ones(num_p + self.chunk_size, dtype=bool), chunk_mask], axis=0
            )

            yt = self.attention_block(chunk_x_with_mem)

            new_mem_params, new_surprise = self.ltm.update_memory(
                yt, prev_long_term_memory, prev_surprise
            )
            Mt_yt = self.ltm.retrieve(yt, new_mem_params)

            return (new_mem_params, new_surprise), (yt, Mt_yt)

        # x_chunks Shape: batch x chunk size x num_chunks x dim
        xs = (x_chunks, mask_chunks)

        init_carry = self.ltm.get_initial_params()
        scan_per_seq = lambda xs: jax.lax.scan(
            forward_update_long_term_memory, init_carry, xs
        )
        _, (yt, Mtyt) = nnx.jit(jax.vmap(scan_per_seq))(xs)

        n_tokens_with_memory = yt.shape[1] * yt.shape[2]

        batched_yt = yt.reshape(batch_size, n_tokens_with_memory, self.d_model)
        batched_mt_yt = Mtyt.reshape(batch_size, n_tokens_with_memory, self.d_model)

        ot = self.gate_norm(batched_yt, batched_mt_yt)

        logits = self._project_to_vocab_dim(ot)

        return logits[:, -self.seq_len :, :]

    @nnx.jit
    def sample_from(self, logits):
        logits, indices = jax.lax.top_k(logits, k=5)
        logits = nnx.softmax(logits)
        return jax.random.choice(jax.random.PRNGKey(0), indices, p=logits)

    @nnx.jit
    def generate_step(self, padded_tokens, sample_index) -> jnp.ndarray:
        logits = self(padded_tokens)
        next_token = self.sample_from(logits[0][sample_index])
        return next_token

    def generate_text(self, tokenizer, max_tokens, start_tokens):
        generated = []
        print(tokenizer.decode(start_tokens), flush=True, end="")
        for i in range(max_tokens):
            sample_index = len(start_tokens) + len(generated) - 1
            padded_tokens = jnp.array(
                (
                    start_tokens
                    + generated
                    + [0] * (self.seq_len - len(start_tokens) - len(generated))
                )
            )[None, :]
            next_token = int(self.generate_step(padded_tokens, sample_index))
            if (
                next_token
                == tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[
                    0
                ]
            ):
                break
            generated.append(next_token)
            print(tokenizer.decode([next_token]), flush=True, end="")
        return tokenizer.decode(start_tokens + generated)
