import jax
import pytest
import typing as t
import jax.numpy as jnp
import flax.linen as nn

from jax import random


def upsample(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    B, H, W, C = x.shape
    x = jax.image.resize(x, (B, H * factor, W * factor, C), method="bilinear")
    return x


class ConvBNRelu(nn.Module):
    out: int
    kernel: int
    running_avg: bool = False
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out, self.kernel, kernel_dilation=self.dilation)(x)
        x = nn.BatchNorm(use_running_average=self.running_avg)(x)
        x = nn.relu(x)

        return x


class RSUBlock(nn.Module):
    levels: int
    out_dim: int
    kernel: int
    mid_dim: int
    running_avg: bool = False

    @nn.compact
    def __call__(self, x):
        down_levels = [
            ConvBNRelu(self.mid_dim, self.kernel, self.running_avg)
            for _ in range(self.levels-1)
        ]


        up_levels = [
            ConvBNRelu(self.mid_dim, self.kernel, self.running_avg)
            for _ in range(self.levels - 1)
        ]


        top_left = ConvBNRelu(self.out_dim, self.kernel, self.running_avg)(x)

        x = top_left
        down_stack = []
        for layer in down_levels:
            x = layer(x)
            down_stack.insert(0, x)
            x = nn.max_pool(x, (2, 2), (2, 2))

        # Insert another convolution without the pooling at the bottom
        down_stack.insert(0, ConvBNRelu(self.mid_dim, self.kernel, self.running_avg)(x))

        x = ConvBNRelu(self.mid_dim, self.kernel, self.running_avg, 2)(x)

        for down, layer in zip(down_stack, up_levels):
            x = jnp.concatenate([down, x], axis=-1)
            x = layer(x)
            x = upsample(x, 2)

        # Final convolution at the top right before concatenation
        x = ConvBNRelu(self.out_dim, self.kernel, self.running_avg)(x)
        out = top_left + x

        return out


class DilationRSUBlock(nn.Module):
    out_dim: int
    kernel: int
    mid_dim: int
    running_avg: bool = False

    @nn.compact
    def __call__(self, x):
        top_left = ConvBNRelu(self.out_dim, self.kernel)(x)

        x = top_left
        d1 = ConvBNRelu(self.mid_dim, self.kernel, self.running_avg)(x)
        d2 = ConvBNRelu(self.mid_dim, self.kernel, self.running_avg)(d1)
        d3 = ConvBNRelu(self.mid_dim, self.kernel, self.running_avg, dilation=2)(d2)
        d4 = ConvBNRelu(self.mid_dim, self.kernel, self.running_avg, dilation=4)(d3)

        b = ConvBNRelu(self.mid_dim, self.kernel, dilation=8)(d4)

        u4 = ConvBNRelu(self.mid_dim, self.kernel, self.running_avg, dilation=4)(jnp.concatenate([d4, b], axis=-1))
        u3 = ConvBNRelu(self.mid_dim, self.kernel, self.running_avg, dilation=4)(jnp.concatenate([d3, u4], axis=-1))
        u2 = ConvBNRelu(self.mid_dim, self.kernel, self.running_avg, dilation=2)(jnp.concatenate([d2, u3], axis=-1))
        u1 = ConvBNRelu(self.out_dim, self.kernel, self.running_avg)(jnp.concatenate([d1, u2], axis=-1))

        out = top_left + u1
        return out


class SideSaliency(nn.Module):
    target_shape: t.Tuple[int, int, int, int]

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(1, (1, 1))(x)
        x = jax.nn.sigmoid(x)
        x = jax.image.resize(x, self.target_shape, method="bilinear")

        return x


class U2Net(nn.Module):
    mid_dim: int = 16
    out_dim: int = 64
    kernel: t.Tuple[int, int] = (3, 3)

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape

        en1 = RSUBlock(7, self.out_dim, self.kernel, self.mid_dim)(x)
        x = nn.max_pool(en1, (2, 2), (2, 2))

        en2 = RSUBlock(6, self.out_dim, self.kernel, self.mid_dim)(x)
        x = nn.max_pool(en2, (2, 2), (2, 2))

        en3 = RSUBlock(5, self.out_dim, self.kernel, self.mid_dim)(x)
        x = nn.max_pool(en3, (2, 2), (2, 2))

        en4 = RSUBlock(4, self.out_dim, self.kernel, self.mid_dim)(x)
        x = nn.max_pool(en4, (2, 2), (2, 2))

        en5 = DilationRSUBlock(self.out_dim, self.kernel, self.mid_dim)(x)

        en6 = DilationRSUBlock(self.out_dim, self.kernel, self.mid_dim)(x)
        sup6 = SideSaliency((B, H, W, C))(en6)

        x = jnp.concatenate([en5, en6], axis=-1)
        x = upsample(x, 2)
        de5 = DilationRSUBlock(self.out_dim, self.kernel, self.mid_dim)(x)
        sup5 = SideSaliency((B, H, W, C))(de5)

        x = jnp.concatenate([de5, en4], axis=-1)
        x = upsample(x, 2)
        de4 = RSUBlock(4, self.out_dim, self.kernel, self.mid_dim)(x)
        sup4 = SideSaliency((B, H, W, C))(de4)

        x = jnp.concatenate([de4, en3], axis=-1)
        x = upsample(x, 2)
        de3 = RSUBlock(5, self.out_dim, self.kernel, self.mid_dim)(x)
        sup3 = SideSaliency((B, H, W, C))(de3)

        x = jnp.concatenate([de3, en2], axis=-1)
        x = upsample(x, 2)
        de2 = RSUBlock(6, self.out_dim, self.kernel, self.mid_dim)(x)
        sup2 = SideSaliency((B, H, W, C))(de2)

        x = jnp.concatenate([de2, en1], axis=-1)
        x = upsample(x, 2)
        de1 = RSUBlock(7, self.out_dim, self.kernel, self.mid_dim)(x)
        sup1 = SideSaliency((B, H, W, C))(de1)

        fused = jnp.concatenate([sup1, sup2, sup3, sup4, sup5, sup6], axis=-1)
        fused = nn.Conv(1, (1, 1))(fused)
        out = jax.nn.sigmoid(fused)

        return out


def test_conv_block():
    out = 5
    kernel = (3,3)

    x = jnp.ones((4, 128, 128, 3))
    layer = ConvBNRelu(out, kernel)
    key = random.PRNGKey(0)
    params = layer.init(key, x)

    y, mutated_vars = layer.apply(params, x, mutable=["batch_stats"])

    assert y.shape == (4, 128, 128, out)
    assert "batch_stats" in mutated_vars.keys()


def test_rsu_block():
    levels = 2
    out_dim= 5
    kernel = (3,3)
    mid_dim = 16

    x = jnp.ones((4, 128, 128, 3))
    block = RSUBlock(levels, out_dim, kernel, mid_dim)
    key = random.PRNGKey(0)
    params = block.init(key, x)

    y, _ = block.apply(params, x, mutable=["batch_stats"])

    assert y.shape == (4, 128, 128, out_dim)


def test_dilation_rsu_block():
    out_dim= 6
    kernel = (3,3)
    mid_dim = 16

    x = jnp.ones((4, 32, 32, 3))
    block = DilationRSUBlock(out_dim, kernel, mid_dim)
    key = random.PRNGKey(0)
    params = block.init(key, x)

    y, _ = block.apply(params, x, mutable=["batch_stats"])

    assert y.shape == (4, 32, 32, out_dim)


def test_upsample():
    x = jnp.ones((2, 32, 32, 3))
    y = upsample(x, 2)

    assert y.shape == (2, 64, 64, 3)


def test_u2_net():
    mid_dim = 16
    out_dim = 64
    kernel = (3, 3)

    x = jnp.ones((4, 256, 256, 3))
    model = U2Net(mid_dim, out_dim, kernel)
    key = random.PRNGKey(0)
    params = model.init(key, x)

    y, _ = model.apply(params, x, mutable=["batch_stats"])
    assert y.shape == (4, 256, 256, 1)

