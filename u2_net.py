import jax
import typing as t

import flax.linen as nn

class ConvBNRelu(nn.Module):
    out: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out, self.kernel_size)(x)
        x = nn.BatchNorm()(x)
        x = nn.relu(x)

        return x


class DownConvBNRelu(nn.Module):
    out: int
    kernel_size: int
    pool_size: t.Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        x = nn.max_pool(x, self.pool_size, self.pool_size)
        x = ConvBNRelu(self.out, self.kernel_size)(x)
        return x


class UpConvBNRelu(nn.Module):
    out: int
    kernel_size: int
    factor: t.Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * self.factor[0], W * self.factor[1], C), method="bilinear")
        x = ConvBNRelu(self.out, self.kernel_size)(x)
        return x


class U2Net(nn.Module):

    @nn.compact
    def __call__(self, x):
        pass


class RSUBlock(nn.Module):
    levels: int
    in_features: int
    out_features: int

    @nn.compact
    def __call__(self, x):
        pass


