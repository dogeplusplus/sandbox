import jax
import pytest
import typing as t
import jax.numpy as jnp
import flax.linen as nn

from jax import random


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


class DilatedConvBN(nn.Module):
    out: int
    kernel: int
    dilation: int
    running_avg: bool = False

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out, self.kernel)(x)
        x = nn.BatchNorm(self.running_avg)(x)
        x = nn.relu(x)

        return x


class U2Net(nn.Module):

    @nn.compact
    def __call__(self, x):
        pass


class RSUBlock(nn.Module):
    levels: int
    out_features: int
    kernel: int
    m: int
    running_avg: bool = False
    dilation: int = 2

    @nn.compact
    def __call__(self, x):
        down_levels = [
            ConvBNRelu(self.m, self.kernel, self.running_avg)
            for _ in range(self.levels-1)
        ]


        up_levels = [
            ConvBNRelu(self.m, self.kernel, self.running_avg)
            for _ in range(self.levels - 1)
        ]


        top_left = ConvBNRelu(self.out_features, self.kernel, self.running_avg)(x)

        x = top_left
        down_stack = []
        for layer in down_levels:
            x = layer(x)
            down_stack.insert(0, x)
            x = nn.max_pool(x, (2, 2), (2, 2))

        # Insert another convolution without the pooling at the bottom
        down_stack.insert(0, ConvBNRelu(self.m, self.kernel, self.running_avg)(x))

        x = ConvBNRelu(self.m, self.kernel, self.running_avg, self.dilation)(x)

        for down, layer in zip(down_stack, up_levels):
            x = jnp.concatenate([down, x], axis=-1)
            x = layer(x)
            B, H, W, C = x.shape
            x = jax.image.resize(x, (B, H * 2, W * 2, C), method="bilinear")

        # Final convolution at the top right before concatenation
        x = ConvBNRelu(self.out_features, self.kernel, self.running_avg)(x)
        out = top_left + x

        return out


class DilationRSUBlock(nn.Module):
    out_features: int
    kernel: int
    m: int
    running_avg: bool = False

    @nn.compact
    def __call__(self, x):
        top_left = ConvBNRelu(self.out_features, self.kernel)(x)

        x = top_left
        d1 = ConvBNRelu(self.m, self.kernel, self.running_avg)(x)
        d2 = ConvBNRelu(self.m, self.kernel, self.running_avg)(d1)
        d3 = ConvBNRelu(self.m, self.kernel, self.running_avg, dilation=2)(d2)
        d4 = ConvBNRelu(self.m, self.kernel, self.running_avg, dilation=4)(d3)

        b = ConvBNRelu(self.m, self.kernel, dilation=8)(d4)

        u4 = ConvBNRelu(self.m, self.kernel, self.running_avg, dilation=4)(jnp.concatenate([d4, b], axis=-1))
        u3 = ConvBNRelu(self.m, self.kernel, self.running_avg, dilation=4)(jnp.concatenate([d3, u4], axis=-1))
        u2 = ConvBNRelu(self.m, self.kernel, self.running_avg, dilation=2)(jnp.concatenate([d2, u3], axis=-1))
        u1 = ConvBNRelu(self.out_features, self.kernel, self.running_avg)(jnp.concatenate([d1, u2], axis=-1))

        out = top_left + u1
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
    out_features= 5
    kernel = (3,3)
    m = 16

    x = jnp.ones((4, 128, 128, 3))
    block = RSUBlock(levels, out_features, kernel, m)
    key = random.PRNGKey(0)
    params = block.init(key, x)

    y, _ = block.apply(params, x, mutable=["batch_stats"])

    assert y.shape == (4, 128, 128, out_features)

def test_dilation_rsu_block():
    out_features= 6
    kernel = (3,3)
    m = 16

    x = jnp.ones((4, 32, 32, 3))
    block = DilationRSUBlock(out_features, kernel, m)
    key = random.PRNGKey(0)
    params = block.init(key, x)

    y, _ = block.apply(params, x, mutable=["batch_stats"])

    assert y.shape == (4, 32, 32, out_features)
