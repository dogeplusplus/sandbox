import jax
import optax
import pickle
import mlflow
import numpy as np
import typing as t
import haiku as hk
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


from tqdm import tqdm
from jax import random
from functools import partial
from argparse import ArgumentParser
from collections import defaultdict
from einops import rearrange, reduce, repeat


class SelfAttention(hk.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads

        self.to_queries = hk.Linear(k*heads, with_bias=False)
        self.to_keys = hk.Linear(k*heads, with_bias=False)
        self.to_values = hk.Linear(k*heads, with_bias=False)
        self.unify_heads = hk.Linear(k)

    def __call__(self, x):
        h = self.heads
        k = self.k

        queries = self.to_queries(x)
        keys = self.to_keys(x)
        values = self.to_values(x)

        queries = rearrange(queries, "b t (k h)  -> (b h) t k", h=h)
        keys = rearrange(keys, "b t (k h) -> (b h) t k", h=h)
        values = rearrange(values, "b t (k h) -> (b h) t k", h=h)

        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))

        dot = jax.lax.batch_matmul(queries, rearrange(keys, "b t k -> b k t"))
        dot = jax.nn.softmax(dot, axis=2)

        out = rearrange(jax.lax.batch_matmul(dot, values), "(b h) t k -> b t (h k)", h=h)
        attention = self.unify_heads(out)

        return attention


class TransformerBlock(hk.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.k = k
        self.heads = heads

        self.attention = SelfAttention(self.k, self.heads)
        self.layer_norm_1 = hk.LayerNorm(axis=[-2, -1], create_scale=True, create_offset=True)
        self.linear_1 = hk.Linear(4*self.k)
        self.linear_2 = hk.Linear(self.k)

        self.layer_norm_2 = hk.LayerNorm(axis=[-2, -1], create_scale=True, create_offset=True)

    def __call__(self, x):
        attended = self.attention(x)
        x = self.layer_norm_1(attended + x)

        forward = self.linear_1(x)
        forward = jax.nn.gelu(forward)
        forward = self.linear_2(forward)

        out = self.layer_norm_2(forward + x)
        return out


def sinusoidal_init(shape, dtype, min_scale=0.1, max_scale=10000.):
    seq_len, d_feature = shape[-2:]
    pe = np.zeros((seq_len, d_feature), dtype=np.float32)
    position = rearrange(np.arange(0, seq_len), "p -> p 1")
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, 0:d_feature:2] = np.sin(position * div_term)
    pe[:, 1:d_feature:2] = np.cos(position * div_term)
    pe = rearrange(pe, "p d -> 1 p d")

    return jnp.array(pe, dtype=dtype)


class SinPosEmb(hk.Module):
    def __init__(self, min_scale=0.1, max_scale=10000.):
        super().__init__()
        self.initializer = partial(sinusoidal_init, min_scale=min_scale, max_scale=max_scale)

    def __call__(self, x):
        pos_emb = hk.get_state("pos_emb", shape=x.shape, dtype=jnp.float32, init=self.initializer)
        return x + pos_emb


class VisionTransformer(hk.Module):
    def __init__(self, k, heads, depth, num_classes, patch_size):
        super().__init__()
        self.k = k
        self.heads = heads
        self.depth = depth
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Patch embedding is just a dense layer mapping a flattened patch to another array
        self.token_emb = hk.Linear(self.k)
        self.blocks = hk.Sequential([
            TransformerBlock(self.k, self.heads) for _ in range(self.depth)
        ])
        self.classification = hk.Linear(self.num_classes)
        self.pos_emb = SinPosEmb()

        self.cls_token = hk.get_parameter("cls", shape=[k], init=hk.initializers.RandomNormal())

        self.classification= hk.Sequential([
            hk.LayerNorm(axis=[-2, -1], create_scale=True, create_offset=True),
            hk.Linear(self.num_classes),
        ])

    def __call__(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        tokens = self.token_emb(x)

        cls_token = repeat(self.cls_token, "k -> b 1 k", b=batch_size)
        combined_tokens = jnp.concatenate([cls_token, tokens], axis=1)

        x = self.pos_emb(combined_tokens)
        x = self.blocks(x)
        x = x[:, 0]
        x = self.classification(x)

        return x


def resize_image(example):
    image = tf.image.resize(example["image"], [32, 32])
    label = example["label"]
    image = tf.image.per_image_standardization(image)

    return image, label


def create_transformer(x):
    return VisionTransformer(
        k=512,
        heads=12,
        depth=2,
        num_classes=100,
        patch_size=4,
    )(x)


def update_metrics(step, metrics, new):
    for name, value in new.items():
        metrics[name] = (metrics[name] * (step - 1) + value) / step

    return metrics


def parse_arguments():
    parser = ArgumentParser("Train Vision Transformer")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    tf.config.set_visible_devices([], 'GPU')

    batch_size = 256
    epochs = 500
    num_classes = 100
    save_every = 10
    show_every = 5

    train_ds, val_ds = tfds.load(
        "cifar100",
        split=["train", "test"],
        shuffle_files=True,
    )

    train_ds = train_ds.map(resize_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(resize_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = tfds.as_numpy(train_ds)
    val_ds = tfds.as_numpy(val_ds)

    model = hk.transform_with_state(create_transformer)
    transformer = hk.without_apply_rng(model)
    xs, _ = next(iter(train_ds))

    rng = random.PRNGKey(42)
    params, state = transformer.init(rng, xs)

    tx = optax.adam(learning_rate=3e-4)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(params, state, xs, ys):
        logits, _ = transformer.apply(params, state, xs)
        one_hot = jax.nn.one_hot(ys, num_classes=num_classes)
        return optax.softmax_cross_entropy(logits, one_hot).sum()

    @jax.jit
    def calculate_metrics(params, state, xs, ys, k=5):
        logits, _ = transformer.apply(params, state, xs)
        classes = logits.argmax(axis=-1)
        accuracy = jnp.mean(classes == ys)

        top_k = np.argsort(logits, axis=-1)[:, -k:]
        hits = (ys == top_k.T).any(axis=0)
        top_k_accuracy = jnp.mean(hits)

        metrics = {
            "accuracy": accuracy,
            f"top_{k}_acc": top_k_accuracy,
        }
        return metrics

    @jax.jit
    def update(
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        xs: tf.Tensor,
        ys: tf.Tensor
    ) -> t.Tuple[hk.Params, optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, state, xs, ys)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state, loss

    if not args.debug:
        mlflow.set_experiment("cifar_haiku")
        mlflow.start_run()

    for e in range(epochs):
        step = 0
        metrics_dict = defaultdict(lambda: 0)
        desc = f"Train Epoch {e}"
        train_bar = tqdm(train_ds, total=len(train_ds), ncols=0, desc=desc)

        for xs, ys in train_bar:
            params, opt_state, loss = update(params, state, opt_state, xs, ys)
            metrics = calculate_metrics(params, state, xs, ys)
            metrics["loss"] = loss

            step += 1
            metrics_dict = update_metrics(step, metrics_dict, metrics)
            if step % show_every == 0:
                metrics_display = {k: round(v, 3) for k, v in metrics_dict.items()}
                train_bar.set_postfix(**metrics_display)

        train_metrics = {f"train_{k}": float(v) for k, v in metrics_dict.items()}
        if not args.debug:
            mlflow.log_metrics(train_metrics, step=e)

        step = 0
        metrics_dict = defaultdict(lambda: 0)
        desc = f"Valid Epoch {e}"
        val_bar = tqdm(val_ds, total=len(val_ds), ncols=0, desc=desc)

        for xs, ys in val_bar:
            loss = loss_fn(params, state, xs, ys)
            metrics = calculate_metrics(params, state, xs, ys)
            metrics["loss"] = loss

            step += 1
            metrics_dict = update_metrics(step, metrics_dict, metrics)
            if step % show_every == 0:
                metrics_display = {k: round(v, 3) for k, v in metrics_dict.items()}
                val_bar.set_postfix(**metrics_display)

        val_metrics = {f"valid_{k}": float(v) for k, v in metrics_dict.items()}

        if not args.debug:
            mlflow.log_metrics(val_metrics, step=e)

        if e % save_every == 0 and not args.debug:
            pickle.dump(params, open("weights.pkl", "wb"))
            mlflow.log_artifact("weights.pkl", "weights")
            pickle.dump(opt_state, open("optimizer.pkl", "wb"))
            mlflow.log_artifact("optimizer.pkl", "optimizer")


if __name__ == "__main__":
    main()
