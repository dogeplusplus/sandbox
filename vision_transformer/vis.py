import jax
import optax
import haiku as hk
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

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
        self.k = k
        self.heads = heads

        self.attention = SelfAttention(self.k, self.heads)
        self.layer_norm_1 = hk.LayerNorm(axis=[-2, -1])
        self.forward = hk.Sequential(
            hk.Linear(4*self.k),
            hk.ReLU(),
            hk.Linear(self.k),
        )
        self.layer_norm_2 = hk.LayerNorm(axis=[-2, -1])


    def __call__(self, x):
        attended = self.attention(x)
        x = self.layer_norm_1(attended + x)
        forward = self.forward(x)
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


        self.pos_emb = hk.Embed(self.seq_length, self.k)
        self.token_emb = hk.Embed(self.num_tokens, self.k)
        self.blocks = hk.Sequential([
            TransformerBlock(self.k, self.heads) for _ in range(self.depth)
        ])
        self.classification = hk.Linear(self.num_classes)

    def __call__(self, x):
        batch_size = x.shape[0]

        positions = jnp.arange(self.seq_length)
        positions = repeat(self.pos_emb(positions), "t k -> b t k", b=batch_size)

        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) p1 p2 c", p1=self.patch_size, p2=self.patch_size)
        tokens = self.token_emb(x)

        x = tokens + positions
        x = self.blocks(x)
        x = self.classification(x)
        x = reduce(x, "b t c -> b c", "mean")

        return jax.nn.log_softmax(x, axis=1)


@jax.jit
def predict(params, image, k, heads, depth, seq_length, num_tokens, num_classes, patch_size):
    transformer = VisionTransformer(
        k,
        heads,
        depth,
        seq_length,
        num_tokens,
        num_classes,
        patch_size,
    )


def main():
    b = 3
    t = 5
    k = 7

    x = jnp.ones((b, t, k))

    def _attention(x):
        layer = SelfAttention(k)
        return layer(x)

    attention = hk.transform(_attention)

    rng = random.PRNGKey(0)
    params = attention.init(rng, x=x)
    y = attention.apply(params=params, x=x, rng=rng)

    batch_size = 16
    epochs = 100

    train_ds, val_ds = tfds.load(
        "food101",
        split=["train", "test"],
        shuffle_files=True,
    )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


    for e in range(epochs):
        for xs, ys in train_ds:
            grads = jax.grad(loss)(params, xs, ys)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)



