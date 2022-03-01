import jax

from jax import random
from einops import rearrange


H = 256
W = 256
C = 3

P = 16
N = H * W / P ** 2

key = random.PRNGKey(42)
image = jax.random.normal(key, (H, W, C))
reshaped = rearrange(image, "(h p1) (w p2) c -> (h w) p1 p2 c", p1=P, p2=P)
