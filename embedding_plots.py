import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from einops import rearrange

def sinusoidal_init(shape, max_len, min_scale=1.0, max_scale=10000.0):
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = rearrange(np.arange(0, max_len), "p -> p 1")
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature:2] = np.sin(position * div_term)
    pe[:, 1:d_feature:2] = np.cos(position * div_term)
    pe = rearrange(pe, "p d -> 1 p d")

    return jnp.array(pe)


def main():
    shape = [100, 200]
    pos_emb = sinusoidal_init(shape, 300)
    plt.matshow(pos_emb[0])
    plt.show()


if __name__ == "__main__":
    main()
