import jax
import optax
import numpy as np
import typing as t
import haiku as hk
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm
from jax import random
from functools import partial
from einops import rearrange, reduce


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

    def __call__(self, x):
        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        tokens = self.token_emb(x)
        x = self.pos_emb(tokens)
        x = self.blocks(x)
        x = self.classification(x)
        x = reduce(x, "b t c -> b c", "mean")

        return jax.nn.log_softmax(x, axis=1)


def resize_image(example):
    image = tf.image.resize(example["image"], [256, 256])
    label = example["label"]
    image = tf.cast(image, tf.float32)

    return image, label


def create_transformer(x):
    return VisionTransformer(
        k=128,
        heads=12,
        depth=768,
        num_classes=101,
        patch_size=32,
    )(x)


def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    batch_size = 64
    epochs = 100
    num_classes = 101

    train_ds, val_ds = tfds.load(
        "food101",
        split=["train", "validation"],
        shuffle_files=True,
    )

    train_ds = train_ds.map(resize_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(resize_image).prefetch(tf.data.AUTOTUNE)
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
    def accuracy(params, state, xs, ys):
        logits, _ = transformer.apply(params, state, xs)
        classes = logits.argmax(axis=-1)
        return jnp.mean(classes == ys)

    @jax.jit
    def update(
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        xs: tf.Tensor,
        ys: tf.Tensor
    ) -> t.Tuple[hk.Params, optax.OptState]:
        grads = jax.grad(loss_fn)(params, state, xs, ys)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state

    show_every = 100
    n = len(train_ds)

    for e in range(epochs):
        step = 0
        train_acc = 0
        train_loss = 0

        desc = f"Train Epoch {e}"
        train_bar = tqdm(train_ds, total=n, ncols=0, desc=desc)
        for xs, ys in train_bar:
            params, opt_state = update(params, state, opt_state, xs, ys)
            loss = loss_fn(params, state, xs, ys)
            acc = accuracy(params, state, xs, ys)

            step += 1
            train_acc *= (step - 1) / step
            train_acc += acc / step
            train_loss *= (step - 1) / step
            train_loss += loss / step

            if step % show_every == 0:
                train_bar.set_postfix(loss=round(train_loss, 3), acc=round(train_acc, 3))


        desc = f"Valid Epoch {e}"
        val_bar = tqdm(train_ds, total=n, ncols=0, desc=desc)
        step = 0
        val_acc = 0
        val_loss = 0
        for xs, ys in val_bar:
            loss = loss_fn(params, state, xs, ys)
            acc = accuracy(params, state, xs, ys)

            step += 1
            val_acc *= (step - 1) / step
            val_acc += acc / step
            val_loss *= (step - 1) / step
            val_loss += loss / step

            if step % show_every == 0:
                val_bar.set_postfix(loss=round(val_loss, 3), acc=round(val_acc, 3))


if __name__ == "__main__":
    main()
