import jax
import optax
import numpy as np
import typing as t
import haiku as hk
import jax.nn as nn
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import trange
from jax import random
from einops import rearrange, repeat, reduce


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
        forward = jax.nn.relu(forward)
        forward = self.linear_2(forward)

        out = self.layer_norm_2(forward + x)
        return out


class VisionTransformer(hk.Module):
    def __init__(self, k, heads, depth, num_tokens, num_classes, patch_size, seq_len):
        super().__init__()
        self.k = k
        self.heads = heads
        self.depth = depth
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.seq_len = seq_len

        # Patch embedding is actually just a dense layer mapping a flattened patch to another array
        self.token_emb = hk.Linear(self.k)
        self.blocks = hk.Sequential([
            TransformerBlock(self.k, self.heads) for _ in range(self.depth)
        ])
        self.classification = hk.Linear(self.num_classes)
        self.pos_emb = hk.Embed(seq_len, self.k)

    def __call__(self, x):
        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        tokens = self.token_emb(x)
        # Position embedding is regular embedding mapping index to (seq_length,) to (seq_length, k)
        positions = repeat(jnp.arange(self.seq_len, dtype=jnp.int32), "t -> 1 t")
        positions = self.pos_emb(positions)

        x = tokens + positions
        x = self.blocks(x)
        x = self.classification(x)
        x = reduce(x, "b t c -> b c", "mean")

        return jax.nn.log_softmax(x, axis=1)


def sinusoidal_init(max_len, min_scale=1.0, max_scale=10000.0):
    def init(key, shape, dtype=np.float32):
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = rearrange(np.arange(0, max_len), "p -> p 1")
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, :d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
        pe = rearrange(pe, "p -> 1 p")

        return jnp.array(pe)

    return init

class SinPosEmb(hk.Module):
    def __call__(self, x):
        embedding = hk.get_parameter("pos_emb", [], init=sinusoidal_init)
        return embedding + x

def main():
    batch_size = 2
    epochs = 100
    num_classes = 101

    train_ds, val_ds = tfds.load(
        "food101",
        split=["train", "validation"],
        shuffle_files=True,
    )
    def resize(example):
        image = tf.image.resize(example["image"], [32, 32])
        label = example["label"]
        image = tf.cast(image, tf.float32)

        return image, label

    train_ds = train_ds.map(resize).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = tfds.as_numpy(train_ds)
    val_ds = tfds.as_numpy(val_ds)

    def create_transformer(x):
        return VisionTransformer(
            k=128,
            heads=8,
            depth=2,
            num_tokens=5,
            num_classes=101,
            patch_size=8,
            seq_len=64,
        )(x)

    model = hk.transform(create_transformer)
    transformer = hk.without_apply_rng(model)

    # xs, _ = next(iter(train_ds))

    xs = jnp.ones((2, 32, 32, 3))

    rng = random.PRNGKey(42)
    params = transformer.init(rng, xs)
    param_count = sum(x.size for x in jax.tree_leaves(params))
    import pdb; pdb.set_trace()

    # tx = optax.adam(learning_rate=3e-4)
    # opt_state = tx.init(params)

    # def loss_fn(params, xs, ys):
    #     logits = transformer.apply(params, xs)
    #     one_hot = jax.nn.one_hot(ys, num_classes=num_classes)
    #     return optax.softmax_cross_entropy(logits, one_hot).sum()

    # @jax.jit
    # def update(
    #     params: hk.Params,
    #     opt_state: optax.OptState,
    #     xs: tf.Tensor,
    #     ys: tf.Tensor
    # ) -> t.Tuple[hk.Params, optax.OptState]:
    #     grads = jax.grad(loss_fn)(params, xs, ys)
    #     updates, opt_state = tx.update(grads, opt_state)
    #     new_params = optax.apply_updates(params, updates)

    #     return new_params, opt_state

    # pbar = trange(epochs)
    # for _ in pbar:
    #     step = 0
    #     for xs, ys in train_ds:
    #         params, opt_state = update(params, opt_state, xs, ys)
    #         pbar.update(step)
    #         step += 1

if __name__ == "__main__":
    main()
