import haiku as hk
import tensorflow as tf
import jax.numpy as jnp

from jax import random

def linear(x):
    return hk.Linear(3)(x)

layer = hk.transform(linear)
x = tf.constant([[1,2,3]], dtype=tf.float32)
rng = random.PRNGKey(42)
params = layer.init(rng, x)
