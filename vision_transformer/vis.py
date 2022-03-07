import jax
import optax
import typing as t
import haiku as hk
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm
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

        queries = rearrange(self.to_queries(x), "b t (k h)  -> (b h) t k", h=h)
        keys = rearrange(self.to_keys(x), "b t (k h) -> (b h) t k", h=h)
        values = rearrange(self.to_values(x), "b t (k h) -> (b h) t k", h=h)

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
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes, patch_size):
        super().__init__()
        self.k = k
        self.heads = heads
        self.depth = depth
        self.seq_length = seq_length
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.patch_size = patch_size

        # Patch embedding is actually just a dense layer mapping a flattened patch to another array
        self.pos_emb = hk.Linear(self.k)
        # Position embedding is regular embedding mapping index to (seq_length,) to (seq_length, k)
        self.token_emb = hk.Embed(self.seq_length, self.k)
        self.blocks = hk.Sequential([
            TransformerBlock(self.k, self.heads) for _ in range(self.depth)
        ])
        self.classification = hk.Linear(self.num_classes)

    def __call__(self, x):
        batch_size = x.shape[0]


        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        tokens = self.token_emb(x)

        positions = repeat(jnp.arange(self.seq_length, dtype="float"), "t -> b t 1", b=batch_size)
        positions = self.pos_emb(positions)
        import pdb; pdb.set_trace()

        x = tokens + positions
        x = self.blocks(x)
        x = self.classification(x)
        x = reduce(x, "b t c -> b c", "mean")

        return jax.nn.log_softmax(x, axis=1)


def main():
    batch_size = 16
    epochs = 100
    num_classes = 101

    train_ds, val_ds = tfds.load(
        "food101",
        split=["train", "validation"],
        shuffle_files=True,
    )

    def resize(example):
        image = tf.image.resize(example["image"], [256, 256])
        label = example["label"]
        image = tf.cast(image, dtype=tf.int32)

        return image, label

    train_ds = train_ds.map(resize).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def create_transformer(x):
        return VisionTransformer(
            k=128,
            heads=8,
            depth=4,
            seq_length=100,
            num_tokens=5,
            num_classes=101,
            patch_size=32,
        )(x)


    model = hk.transform(create_transformer)
    transformer = hk.without_apply_rng(model)

    xs, _ = next(iter(train_ds))
    rng = random.PRNGKey(42)
    params = transformer.init(rng, xs)
    tx = optax.adam(lr=3e-4)
    opt_state = tx.init(params)

    def loss_fn(params, xs, ys):
        logits = transformer.apply(params, xs)
        one_hot = jax.nn.one_hot(ys, num_classes=num_classes)
        return optax.softmax_cross_entropy(logits, one_hot).sum()

    @jax.jit
    def update(
        params: hk.Params,
        opt_state: optax.OptState,
        xs: tf.Tensor,
        ys: tf.Tensor
    ) -> t.Tuple[hk.Params, optax.OptState]:
        grads = jax.grad(loss_fn)(params, xs, ys)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state

    for _ in tqdm.trange(epochs):
        for xs, ys in train_ds:
            params, opt_state = update(params, opt_state, xs, ys)


if __name__ == "__main__":
    main()
