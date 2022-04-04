import jax
import typing as t
import jax.numpy as jnp

import flax.linen as nn


class ConvBNRelu(nn.Module):
    out: int
    kernel_size: int
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out, self.kernel_size, dilation=self.dilation)(x)
        x = nn.BatchNorm()(x)
        x = nn.relu(x)

        return x


class DownSample(nn.Module):
    out: int
    kernel_size: int
    pool_size: t.Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        x = nn.max_pool(x, self.pool_size, self.pool_size)
        x = ConvBNRelu(self.out, self.kernel_size)(x)
        return x


class UpSample(nn.Module):
    out: int
    kernel_size: int
    factor: t.Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * self.factor[0], W * self.factor[1], C),
                             method="bilinear")
        x = ConvBNRelu(self.out, self.kernel_size)(x)
        return x


class DilatedConvBN(nn.Module):
    out: int
    kernel_size: int
    dilation: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out, self.kernel_size)(x)
        x = nn.BatchNorm()(x)
        x = nn.relu(x)

        return x


class U2Net(nn.Module):

    @nn.compact
    def __call__(self, x):
        pass


class RSUBlock(nn.Module):
    levels: int
    in_features: int
    out_features: int
    kernel_size: int
    m: int

    @nn.compact
    def __call__(self, x):
        top_left = ConvBNRelu(self.out_features, self.kernel_size)(x)

        down_levels = [ConvBNRelu(self.m, self.kernel_size)] + [
            DownSample(self.m, self.kernel_size, (2, 2))
            for _ in range(self.levels - 2)
        ]

        bottom = ConvBNRelu(self.m, self.kernel_size, self.dilation)

        up_levels = [ConvBNRelu(self.m, self.kernel_size)] + [
            UpSample(self.m, self.kernel_size, (2, 2))
            for _ in range(self.levels - 2)
        ] + [UpSample(self.out_features, self.kernel_size, (2, 2))]


        down_stack = []
        for layer in down_levels:
            x = layer(x)
            down_stack.insert(0, x)

        x = bottom(x)

        for down, layer in enumerate(up_levels):
            x = jnp.concatenate([down, x], axis=1)
            x = layer(x)

        out = top_left + x

        return out

