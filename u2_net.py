import jax
import optax
import typing as t
import jax.numpy as jnp
import flax.linen as nn
import tensorflow as tf

from pathlib import Path
from einops import repeat
from tqdm import tqdm
from jax import random
from collections import defaultdict
from flax.core.frozen_dict import FrozenDict


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
        B, H, W, _ = x.shape

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
        sup6 = SideSaliency((B, H, W, 1))(en6)

        x = jnp.concatenate([en5, en6], axis=-1)
        x = upsample(x, 2)
        de5 = DilationRSUBlock(self.out_dim, self.kernel, self.mid_dim)(x)
        sup5 = SideSaliency((B, H, W, 1))(de5)

        x = jnp.concatenate([de5, en4], axis=-1)
        x = upsample(x, 2)
        de4 = RSUBlock(4, self.out_dim, self.kernel, self.mid_dim)(x)
        sup4 = SideSaliency((B, H, W, 1))(de4)

        x = jnp.concatenate([de4, en3], axis=-1)
        x = upsample(x, 2)
        de3 = RSUBlock(5, self.out_dim, self.kernel, self.mid_dim)(x)
        sup3 = SideSaliency((B, H, W, 1))(de3)

        x = jnp.concatenate([de3, en2], axis=-1)
        x = upsample(x, 2)
        de2 = RSUBlock(6, self.out_dim, self.kernel, self.mid_dim)(x)
        sup2 = SideSaliency((B, H, W, 1))(de2)

        x = jnp.concatenate([de2, en1], axis=-1)
        x = upsample(x, 2)
        de1 = RSUBlock(7, self.out_dim, self.kernel, self.mid_dim)(x)
        sup1 = SideSaliency((B, H, W, 1))(de1)

        fused = jnp.concatenate([sup1, sup2, sup3, sup4, sup5, sup6], axis=-1)
        fused = nn.Conv(1, (1, 1))(fused)
        out = jax.nn.sigmoid(fused)

        return jnp.concatenate([out, sup1, sup2, sup3, sup4, sup5, sup6], axis=-1)


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

    saliency_maps, _ = model.apply(params, x, mutable=["batch_stats"])
    assert saliency_maps.shape == (4, 256, 256, 7)
    assert jnp.max(saliency_maps) <= 1
    assert jnp.min(saliency_maps) >= 0


def test_saliency_map():
    target_shape = (3, 64, 64, 3)
    x = jnp.ones((3, 8, 8, 3))

    layer = SideSaliency(target_shape)
    key = random.PRNGKey(0)
    params = layer.init(key, x)
    saliency_map = layer.apply(params, x)
    assert saliency_map.shape == target_shape
    assert jnp.max(saliency_map) <= 1
    assert jnp.min(saliency_map) >= 0


def normalize_image(img, a=-1, b=1):
    lower = tf.reduce_min(img)
    upper = tf.reduce_max(img)
    img = (img - lower) / (upper - lower)
    img = a + (b - a) * img
    return img


def parse_image(filename, channels=3):
    image = tf.io.read_file(filename)
    image = tf.io.decode_image(image, channels, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [320, 320])
    return image


def duts_dataset(img_dir: Path, label_dir: Path, batch_size: int, val_ratio: float = 0.2):
    images = tf.data.Dataset.list_files(str(img_dir / "*"))
    labels = images.map(lambda x: tf.strings.regex_replace(x, str(img_dir), str(label_dir)))


    images = images.map(parse_image)
    images = images.map(lambda x: normalize_image(x, -1, 1))

    labels = labels.map(lambda x: tf.strings.regex_replace(x, ".jpg", ".png"))
    labels = labels.map(lambda x: parse_image(x, 1))
    labels = labels.map(lambda x: normalize_image(x, 0, 1))

    ds = tf.data.Dataset.zip((images, labels))
    size = len(ds)
    val_size = int(size * val_ratio)

    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


if __name__ == "__main__":
    img_dir = Path("..", "..", "Downloads", "DUTS-TR", "DUTS-TR-Image")
    label_dir = Path("..", "..", "Downloads", "DUTS-TR", "DUTS-TR-Mask")

    tf.config.set_visible_devices([], "GPU")
    writer = tf.summary.create_file_writer("logs")

    samples = 8
    batch_size = 16
    train_ds, val_ds = duts_dataset(img_dir, label_dir, batch_size)

    epochs = 1
    mid_dim = 16
    out_dim = 64
    kernel = (3, 3)

    x = jnp.zeros((2, 320, 320, 3))
    model = U2Net(mid_dim, out_dim, kernel)
    key = random.PRNGKey(0)
    params = model.init(key, x)

    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(
        params: FrozenDict,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
        weights: jnp.ndarray = jnp.ones(7),
    ) -> jnp.ndarray:
        saliency_maps, _ = model.apply(params, xs, mutable=["batch_stats"])
        ys = repeat(ys, "b h w 1 -> b h w x", x=len(weights))
        losses = optax.sigmoid_binary_cross_entropy(saliency_maps, ys)
        total_loss = jnp.mean(weights * losses)

        return total_loss

    @jax.jit
    def update(
        params: FrozenDict,
        opt_state: optax.OptState,
        xs: jnp.ndarray,
        ys: jnp.ndarray,
    ) -> t.Tuple[FrozenDict, optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state, loss

    for e in range(epochs):
        step = 0
        metrics_dict = defaultdict(lambda: 0)
        desc = f"Train Epoch {e}"
        train_bar = tqdm(train_ds, total=len(train_ds), ncols=0, desc=desc)
        for xs, ys in train_bar:
            xs = jnp.asarray(xs)
            ys = jnp.asarray(ys)

            params, opt_state, loss = update(params, opt_state, xs, ys)
            metrics_dict["loss"] = (step * metrics_dict["loss"] + loss) / (step + 1)

            train_bar.set_postfix(**metrics_dict)
            step += 1

