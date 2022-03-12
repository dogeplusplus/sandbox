import jax
import haiku as hk
import jax.numpy as jnp

import tensorflow as tf
import tensorflow_datasets as tfds

class SelfAttention(hk.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads

        self.to_queries = hk.Linear(k*heads, with_bias=False)

    def __call__(self, x):
        return self.to_queries(x)


def attn(x):
    return SelfAttention(8, 8)(x)

layer = hk.transform(attn)
rng = jax.random.PRNGKey(42)
x = jnp.ones((32, 8))
params = layer.init(rng, x)

import pdb; pdb.set_trace()



