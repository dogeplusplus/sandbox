import jax
import optax
import pickle
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


class ConvLNRelu(nn.Module):
    out: int
    kernel: int
    inference: bool = False
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out, self.kernel, kernel_dilation=self.dilation)(x)
        # x = nn.BatchNorm(use_running_average=self.inference)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        return x


class RSUBlock(nn.Module):
    levels: int
    out: int
    kernel: int
    mid: int
    inference: bool = False

    @nn.compact
    def __call__(self, x):
        down_levels = [
            ConvLNRelu(self.mid, self.kernel, self.inference)
            for _ in range(self.levels - 1)
        ]

        up_levels = [
            ConvLNRelu(self.mid, self.kernel, self.inference)
            for _ in range(self.levels - 1)
        ]

        top_left = ConvLNRelu(self.out, self.kernel, self.inference)(x)

        x = top_left
        down_stack = []
        for layer in down_levels:
            x = layer(x)
            down_stack.insert(0, x)
            x = nn.max_pool(x, (2, 2), (2, 2))

        # Insert another convolution without the pooling at the bottom
        down_stack.insert(0, ConvLNRelu(self.mid, self.kernel, self.inference)(x))

        x = ConvLNRelu(self.mid, self.kernel, self.inference, 2)(x)

        for down, layer in zip(down_stack, up_levels):
            x = jnp.concatenate([down, x], axis=-1)
            x = layer(x)
            x = upsample(x, 2)

        # Final convolution at the top right before concatenation
        x = ConvLNRelu(self.out, self.kernel, self.inference)(x)
        out = top_left + x

        return out


class DilationRSUBlock(nn.Module):
    out: int
    kernel: int
    mid: int
    inference: bool = False

    @nn.compact
    def __call__(self, x):
        top_left = ConvLNRelu(self.out, self.kernel, self.inference)(x)

        x = top_left
        d1 = ConvLNRelu(self.mid, self.kernel, self.inference)(x)
        d2 = ConvLNRelu(self.mid, self.kernel, self.inference)(d1)
        d3 = ConvLNRelu(self.mid, self.kernel, self.inference, dilation=2)(d2)
        d4 = ConvLNRelu(self.mid, self.kernel, self.inference, dilation=4)(d3)

        b = ConvLNRelu(self.mid, self.kernel, self.inference, dilation=8)(d4)

        u4 = ConvLNRelu(self.mid, self.kernel, self.inference, dilation=4)(
            jnp.concatenate([d4, b], axis=-1)
        )
        u3 = ConvLNRelu(self.mid, self.kernel, self.inference, dilation=4)(
            jnp.concatenate([d3, u4], axis=-1)
        )
        u2 = ConvLNRelu(self.mid, self.kernel, self.inference, dilation=2)(
            jnp.concatenate([d2, u3], axis=-1)
        )
        u1 = ConvLNRelu(self.out, self.kernel, self.inference)(
            jnp.concatenate([d1, u2], axis=-1)
        )

        out = top_left + u1
        return out


class SideSaliency(nn.Module):
    target_shape: t.Tuple[int, int, int, int]

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(1, (3, 3))(x)
        x = jax.image.resize(x, self.target_shape, method="bilinear")
        x = jax.nn.sigmoid(x)

        return x


class U2Net(nn.Module):
    mid: int = 16
    out: int = 64
    kernel: t.Tuple[int, int] = (3, 3)
    inference: bool = False

    @nn.compact
    def __call__(self, x):
        B, H, W, _ = x.shape

        en1 = RSUBlock(7, self.out, self.kernel, self.mid, self.inference)(x)
        x = nn.max_pool(en1, (2, 2), (2, 2))

        en2 = RSUBlock(6, self.out, self.kernel, self.mid, self.inference)(x)
        x = nn.max_pool(en2, (2, 2), (2, 2))

        en3 = RSUBlock(5, self.out, self.kernel, self.mid, self.inference)(x)
        x = nn.max_pool(en3, (2, 2), (2, 2))

        en4 = RSUBlock(4, self.out, self.kernel, self.mid, self.inference)(x)
        x = nn.max_pool(en4, (2, 2), (2, 2))

        en5 = DilationRSUBlock(self.out, self.kernel, self.mid, self.inference)(
            x
        )

        en6 = DilationRSUBlock(self.out, self.kernel, self.mid, self.inference)(
            x
        )
        sup6 = SideSaliency((B, H, W, 1))(en6)

        x = jnp.concatenate([en5, en6], axis=-1)
        x = upsample(x, 2)
        de5 = DilationRSUBlock(self.out, self.kernel, self.mid, self.inference)(
            x
        )
        sup5 = SideSaliency((B, H, W, 1))(de5)

        x = jnp.concatenate([de5, en4], axis=-1)
        x = upsample(x, 2)
        de4 = RSUBlock(4, self.out, self.kernel, self.mid, self.inference)(x)
        sup4 = SideSaliency((B, H, W, 1))(de4)

        x = jnp.concatenate([de4, en3], axis=-1)
        x = upsample(x, 2)
        de3 = RSUBlock(5, self.out, self.kernel, self.mid, self.inference)(x)
        sup3 = SideSaliency((B, H, W, 1))(de3)

        x = jnp.concatenate([de3, en2], axis=-1)
        x = upsample(x, 2)
        de2 = RSUBlock(6, self.out, self.kernel, self.mid, self.inference)(x)
        sup2 = SideSaliency((B, H, W, 1))(de2)

        x = jnp.concatenate([de2, en1], axis=-1)
        x = upsample(x, 2)
        de1 = RSUBlock(7, self.out, self.kernel, self.mid, self.inference)(x)
        sup1 = SideSaliency((B, H, W, 1))(de1)

        fused = jnp.concatenate([sup1, sup2, sup3, sup4, sup5, sup6], axis=-1)
        fused = nn.Conv(1, (1, 1))(fused)
        out = jax.nn.sigmoid(fused)

        return jnp.concatenate([out, sup1, sup2, sup3, sup4, sup5, sup6], axis=-1)


def test_conv_block():
    out = 5
    kernel = (3, 3)

    x = jnp.ones((4, 128, 128, 3))
    layer = ConvLNRelu(out, kernel)
    key = random.PRNGKey(0)
    params = layer.init(key, x)

    y, mutated_vars = layer.apply(params, x, mutable=["batch_stats"])

    assert y.shape == (4, 128, 128, out)
    assert "batch_stats" in mutated_vars.keys()


def test_rsu_block():
    levels = 2
    out = 5
    kernel = (3, 3)
    mid = 16

    x = jnp.ones((4, 128, 128, 3))
    block = RSUBlock(levels, out, kernel, mid)
    key = random.PRNGKey(0)
    params = block.init(key, x)

    y, _ = block.apply(params, x, mutable=["batch_stats"])

    assert y.shape == (4, 128, 128, out)


def test_dilation_rsu_block():
    out = 6
    kernel = (3, 3)
    mid = 16

    x = jnp.ones((4, 32, 32, 3))
    block = DilationRSUBlock(out, kernel, mid)
    key = random.PRNGKey(0)
    params = block.init(key, x)

    y, _ = block.apply(params, x, mutable=["batch_stats"])

    assert y.shape == (4, 32, 32, out)


def test_upsample():
    x = jnp.ones((2, 32, 32, 3))
    y = upsample(x, 2)

    assert y.shape == (2, 64, 64, 3)


def test_u2_net():
    mid = 16
    out = 64
    kernel = (3, 3)

    x = jnp.ones((4, 256, 256, 3))
    model = U2Net(mid, out, kernel)
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


def duts_dataset(
    img_dir: Path,
    label_dir: Path,
    batch_size: int,
    val_ratio: float = 0.2,
    shuffle_buffer: int = 8,
):
    images = tf.data.Dataset.list_files(str(img_dir / "*"), shuffle=False)
    labels = images.map(
        lambda x: tf.strings.regex_replace(x, str(img_dir), str(label_dir))
    )

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

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])
    train_ds = train_ds.map(augmentation)

    train_ds = (
        train_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


    val_ds = (
        val_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds

def duts_dataset_original_paper(
    img_dir: Path,
    label_dir: Path,
    batch_size: int,
    val_ratio: float = 0.2,
    shuffle_buffer: int = 8,
):

    norm = lambda x: (x - 0.485) / 0.229

    images = tf.data.Dataset.list_files(str(img_dir / "*"), shuffle=False)
    labels = images.map(
        lambda x: tf.strings.regex_replace(x, str(img_dir), str(label_dir))
    )

    images = images.map(parse_image)
    images = images.map(lambda x: norm(x))

    labels = labels.map(lambda x: tf.strings.regex_replace(x, ".jpg", ".png"))
    labels = labels.map(lambda x: parse_image(x, 1))
    labels = labels.map(lambda x: normalize_image(x, 0, 1))

    ds = tf.data.Dataset.zip((images, labels))
    size = len(ds)
    val_size = int(size * val_ratio)

    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size)

    train_ds = (
        train_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds

def bce_loss(preds: jnp.ndarray, labels: jnp.ndarray) -> float:
    EPS = 1e-8
    preds = jnp.clip(preds, EPS, 1 - EPS)
    loss = -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds + EPS))
    return loss


def main():
    img_dir = Path("..", "..", "Downloads", "DUTS-TR", "DUTS-TR-Image")
    label_dir = Path("..", "..", "Downloads", "DUTS-TR", "DUTS-TR-Mask")

    tf.config.set_visible_devices([], "GPU")
    train_writer = tf.summary.create_file_writer("logs/train")
    valid_writer = tf.summary.create_file_writer("logs/valid")

    batch_size = 8
    train_ds, val_ds = duts_dataset(img_dir, label_dir, batch_size)
    sample_train_img, sample_train_lab = next(iter(train_ds))
    sample_val_img, sample_val_lab = next(iter(val_ds))

    epochs = 100
    mid = 16
    out = 64
    kernel = (3, 3)
    log_every = 1

    x = jnp.zeros((2, 320, 320, 3))
    model = U2Net(mid, out, kernel)
    inference_model = U2Net(mid, out, kernel, True)
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
        losses = bce_loss(saliency_maps, ys)
        total_loss = jnp.mean(weights * losses)

        return total_loss

    @jax.jit
    def update(
        params: FrozenDict, opt_state: optax.OptState, xs: jnp.ndarray, ys: jnp.ndarray,
    ) -> t.Tuple[FrozenDict, optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state, loss

    with train_writer.as_default():
        tf.summary.image("images", sample_train_img, step=0, max_outputs=8)
        tf.summary.image("labels", sample_train_lab, step=0, max_outputs=8)

    with valid_writer.as_default():
        tf.summary.image("images", sample_val_img, step=0, max_outputs=8)
        tf.summary.image("labels", sample_val_lab, step=0, max_outputs=8)

    train_jax_img = jnp.asarray(sample_train_img)
    val_jax_img = jnp.asarray(sample_val_img)

    for e in range(epochs):
        train_step = 0
        val_step = 0
        train_metrics = defaultdict(lambda: 0)
        val_metrics = defaultdict(lambda: 0)
        train_bar = tqdm(train_ds, total=len(train_ds), ncols=0, desc=f"Train Epoch {e}")
        val_bar = tqdm(val_ds, total=len(val_ds), ncols=0, desc=f"Valid Epoch {e}")

        for xs, ys in train_bar:
            xs = jnp.asarray(xs)
            ys = jnp.asarray(ys)

            params, opt_state, loss = update(params, opt_state, xs, ys)
            train_metrics["loss"] = (train_step * train_metrics["loss"] + loss) / (train_step + 1)

            train_bar.set_postfix(**train_metrics)
            train_step += 1

        for xs, ys in val_bar:
            xs = jnp.asarray(xs)
            ys = jnp.asarray(ys)

            loss = loss_fn(params, xs, ys)
            val_metrics["loss"] = (val_step * val_metrics["loss"] + loss) / (
                val_step + 1
            )

            val_bar.set_postfix(**val_metrics)
            val_step += 1

        with train_writer.as_default():
            tf.summary.scalar("loss", train_metrics["loss"], step=e)
        with valid_writer.as_default():
            tf.summary.scalar("loss", val_metrics["loss"], step=e)

        if e % log_every == 0:
            pickle.dump(params, open("weights.pkl", "wb"))
            pickle.dump(opt_state, open("optimizer.pkl", "wb"))

            with train_writer.as_default():
                pred_train = inference_model.apply(params, train_jax_img)[..., [0]]
                tf.summary.image("predictions", pred_train, step=e, max_outputs=8)

            with valid_writer.as_default():
                pred_val = inference_model.apply(params, val_jax_img)[..., [0]]
                tf.summary.image("predictions", pred_val, step=e, max_outputs=8)

        train_writer.flush()
        valid_writer.flush()


if __name__ == "__main__":
    main()
