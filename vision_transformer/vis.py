import jax
import haiku as hk
from jax import random
import jax.numpy as jnp
from einops import rearrange


H = 256
W = 256
C = 3

P = 16
N = H * W / P ** 2

key = random.PRNGKey(42)
image = jax.random.normal(key, (H, W, C))
reshaped = rearrange(image, "(h p1) (w p2) c -> (h w) p1 p2 c", p1=P, p2=P)

class SelfAttention(hk.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads

    def __call__(self, x):
        h = self.heads

        queries = rearrange(hk.Linear(k*h, with_bias=False)(x), "b t (k h)  -> (b h) t k", h=h)
        keys = rearrange(hk.Linear(k*h, with_bias=False)(x), "b t (k h) -> (b h) t k", h=h)
        values = rearrange(hk.Linear(k*h, with_bias=False)(x), "b t (k h) -> (b h) t k", h=h)

        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))

        dot = jax.lax.batch_matmul(queries, rearrange(keys, "b t k -> b k t"))
        dot = jax.nn.softmax(dot, axis=2)

        out = rearrange(jax.lax.batch_matmul(dot, values), "(b h) t k -> b t (h k)", h=h)
        unify_heads = hk.Linear(k)(out)

        return unify_heads

class TransformerBlock(hk.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        self.k = k
        self.heads = heads
        self.depth = depth
        self.seq_length = seq_length
        self.num_tokens = num_tokens
        self.num_classes = num_classes

    def __call__(self, x):
        token_emb = hk.Embed()
        pos_emb = hk.Embed()


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
import pdb; pdb.set_trace()
