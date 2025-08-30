from __future__ import annotations
import argparse
import jax, jax.numpy as jnp
from flax import nnx
import numpy as np
import tiktoken

from variants.mac import TitanMAC
from jax.sharding import NamedSharding, PartitionSpec
from data_handler.loader import load_dataset
from training.losses import cross_entropy
import optax
import time
from configs.mac_config import MacConfig
from training._mesh import create_mesh


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    grad_fn = nnx.value_and_grad(cross_entropy, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    jax.debug.print("Training loss: {loss}", loss=loss)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(grads)


def get_optimizer(model: nnx.Module, cfg: MacConfig) -> nnx.Optimizer:
    schedule = optax.cosine_decay_schedule(
        init_value=cfg.optimizer.opt_learning_rate, decay_steps=cfg.training.max_steps
    )
    optax_chain = optax.chain(
        optax.adamw(learning_rate=schedule, weight_decay=cfg.optimizer.opt_weight_decay)
    )
    return nnx.Optimizer(model, optax_chain, wrt=nnx.Param)


def create_model(cfg: MacConfig, n_vocab: int, rngs: nnx.Rngs) -> nnx.Module:
    model_cfg = cfg.model
    d_model = model_cfg.d_model
    chunk_size = model_cfg.chunk_size
    n_heads = model_cfg.n_heads
    seq_len = model_cfg.seq_len
    n_persist = model_cfg.n_persist
    n_mem_tokens = model_cfg.n_mem_tokens
    mem_hidden = model_cfg.mem_hidden

    return TitanMAC(
        d_model=d_model,
        n_heads=n_heads,
        seq_len=seq_len,
        chunk_size=chunk_size,
        n_persist=n_persist,
        n_mem_tokens=n_mem_tokens,
        mem_hidden=mem_hidden,
        vocab_size=n_vocab,
        rngs=rngs,
    )


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", type=str, default="configs/mac.yaml")
    ap.add_argument(
        "--train_data_path",
        type=str,
        default="DEFAULT_PATH",
        help="Path to the train dataset",
    )
    ap.add_argument(
        "--val_data_path",
        type=str,
        default="DEFAULT_PATH",
        help="Path to the val dataset",
    )
    return ap.parse_args()


def get_batch(
    data: np.ndarray, seq_len: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    ix = np.random.randint(0, len(data) - seq_len, (batch_size,))
    x = np.stack([(data[i : i + seq_len]).astype(jnp.int64) for i in ix])
    y = np.stack([(data[i + 1 : i + 1 + seq_len]).astype(jnp.int64) for i in ix])
    return x, y


def main():
    args = get_args()
    py_main(args.config_path, args.train_data_path, args.val_data_path)


def py_main(config_path: str, train_data_path: str, val_data_path: str):

    cfg = MacConfig.from_yaml(config_path)
    # TODO: Make it configurable
    tokenizer = tiktoken.get_encoding("gpt2")

    train_data = load_dataset(train_data_path)
    val_data = load_dataset(val_data_path)

    max_steps = cfg.training.max_steps

    rngs = nnx.Rngs(cfg.general.seed)
    seq_len = cfg.model.seq_len
    batch_size = cfg.training.batch_size
    model = create_model(cfg, tokenizer.n_vocab, rngs)
    optimizer = get_optimizer(model, cfg)
    train_metrics = nnx.metrics.Average("loss")
    val_metrics = nnx.metrics.Average("val_loss")
    metrics_history = {"train_loss": [], "val_loss": []}

    mesh = create_mesh()
    step = 0
    start_time = time.time()
    while True:
        input_batch, target_batch = get_batch(train_data, seq_len, batch_size)
        batch = (input_batch, target_batch)
        if len(input_batch) % len(jax.devices()) != 0:
            continue
        if mesh:
            batch = jax.device_put(
                batch, NamedSharding(mesh, PartitionSpec("batch", None))
            )
        train_step(model, optimizer, train_metrics, batch)

        if step % 200 == 0:
            train_loss = float(train_metrics.compute())
            metrics_history["train_loss"].append(train_loss)

            elapsed_time = time.time() - start_time
            print(
                f"Step {step + 1}, Training loss: {train_loss}, Elapsed Time: {elapsed_time:.2f} seconds"
            )

            input_val_batch, target_val_batch = get_batch(val_data, seq_len, batch_size)
            eval_batch = (input_val_batch, target_val_batch)

            if mesh:
                eval_batch = jax.device_put(
                    eval_batch, NamedSharding(mesh, PartitionSpec("batch", None))
                )

            loss, logits = cross_entropy(model, eval_batch)
            val_metrics.update(val_loss=loss, logits=logits)
            val_loss = float(val_metrics.compute())
            metrics_history["val_loss"].append(val_loss)
            print(f"Step {step + 1}, Validation loss: {val_loss}")
            train_metrics.reset()
            val_metrics.reset()

            start_time = time.time()
        step += 1

        if step > max_steps:
            break


if __name__ == "__main__":
    main()
