import jax
import optax
import pickle
import mlflow
import typing as t
import haiku as hk
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from einops import rearrange, repeat


class SelfAttention(hk.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads

        self.to_queries = hk.Linear(k*heads, with_bias=False)
        self.to_keys = hk.Linear(k*heads, with_bias=False)
        self.to_values = hk.Linear(k*heads, with_bias=False)
        self.unify_heads = hk.Linear(k)

    def __call__(self, x):
        h = self.heads
        k = self.k

        queries = self.to_queries(x)
        keys = self.to_keys(x)
        values = self.to_values(x)

        queries = rearrange(queries, "b t (k h)  -> (b h) t k", h=h)
        keys = rearrange(keys, "b t (k h) -> (b h) t k", h=h)
        values = rearrange(values, "b t (k h) -> (b h) t k", h=h)

        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))

        dot = jax.lax.batch_matmul(queries, rearrange(keys, "b t k -> b k t"))
        dot = jax.nn.softmax(dot, axis=2)

        out = rearrange(jax.lax.batch_matmul(dot, values), "(b h) t k -> b t (h k)", h=h)
        attention = self.unify_heads(out)

        return attention


class TransformerBlock(hk.Module):
    def __init__(self, k, heads, dropout):
        super().__init__()
        self.k = k
        self.heads = heads
        self.dropout = dropout

        self.attention = SelfAttention(self.k, self.heads)
        self.layer_norm_1 = hk.LayerNorm(axis=[-2, -1], create_scale=True, create_offset=True)
        self.linear_1 = hk.Linear(4*self.k)
        self.linear_2 = hk.Linear(self.k)

        self.layer_norm_2 = hk.LayerNorm(axis=[-2, -1], create_scale=True, create_offset=True)

    def __call__(self, x, inference=False):
        dropout = 0. if inference else self.dropout

        attended = self.attention(x)
        x = self.layer_norm_1(attended + x)

        key1 = hk.next_rng_key()
        key2 = hk.next_rng_key()

        forward = self.linear_1(x)
        forward = jax.nn.gelu(forward)
        forward = hk.dropout(key1, dropout, forward)
        forward = self.linear_2(forward)
        forward = self.layer_norm_2(forward + x)
        out = hk.dropout(key2, dropout, forward)

        return out


class VisionTransformer(hk.Module):
    def __init__(self, k, heads, depth, num_classes, patch_size, image_size, dropout):
        super().__init__()
        self.k = k
        self.heads = heads
        self.depth = depth
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.image_size = image_size
        self.dropout = dropout

        # Patch embedding is just a dense layer mapping a flattened patch to another array
        self.token_emb = hk.Linear(self.k)
        self.blocks = hk.Sequential([
            TransformerBlock(self.k, self.heads, dropout) for _ in range(self.depth)
        ])
        self.classification = hk.Linear(self.num_classes)
        height, width = image_size
        num_patches = (height // patch_size) * (width // patch_size) + 1

        self.pos_emb = hk.Embed(vocab_size=num_patches, embed_dim=self.k)
        self.cls_token = hk.get_parameter("cls", shape=[k], init=hk.initializers.RandomNormal())

        self.classification= hk.Sequential([
            hk.LayerNorm(axis=[-2, -1], create_scale=True, create_offset=True),
            hk.Linear(self.num_classes),
        ])

    def __call__(self, x, inference=False):
        dropout = 0. if inference else self.dropout

        batch_size = x.shape[0]
        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        tokens = self.token_emb(x)

        cls_token = repeat(self.cls_token, "k -> b 1 k", b=batch_size)
        combined_tokens = jnp.concatenate([cls_token, tokens], axis=1)
        positions = jnp.arange(combined_tokens.shape[1])
        pos_emb = self.pos_emb(positions)
        x = pos_emb + combined_tokens
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = self.blocks(x)
        x = x[:, 0]
        x = self.classification(x)

        return x


def resize_image(example):
    image = tf.image.resize(example["image"], [72, 72])
    label = example["label"]
    image = tf.image.per_image_standardization(image)

    return image, label


def update_metrics(step, metrics, new):
    for name, value in new.items():
        metrics[name] = (metrics[name] * (step - 1) + value) / step

    return metrics


def parse_arguments():
    parser = ArgumentParser("Train Vision Transformer")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    tf.config.set_visible_devices([], 'GPU')

    batch_size = 256
    epochs = 500
    num_classes = 100
    save_every = 10
    show_every = 5
    k = 256
    heads = 4
    depth = 8
    patch_size = 6
    image_size = (72, 72)
    dropout = 0.2

    def create_transformer(x):
        return VisionTransformer(
            k,
            heads,
            depth,
            num_classes,
            patch_size,
            image_size,
            dropout,
        )(x)

    dataset_name = "cifar100"
    train_ds, val_ds = tfds.load(
        dataset_name,
        split=["train", "test"],
        shuffle_files=True,
    )
    train_ds = train_ds.map(resize_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(resize_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = tfds.as_numpy(train_ds)
    val_ds = tfds.as_numpy(val_ds)

    transformer = hk.transform(create_transformer)
    xs, _ = next(iter(train_ds))

    rng_seq = hk.PRNGSequence(42)
    params = transformer.init(next(rng_seq), xs)
    decay_steps = 10000
    lr_scheduler = optax.cosine_decay_schedule(1e-3, decay_steps)
    tx = optax.adam(lr_scheduler)
    opt_state = tx.init(params)


    @jax.jit
    def loss_fn(params, key, xs, ys):
        logits = transformer.apply(params, key, xs)
        one_hot = jax.nn.one_hot(ys, num_classes=num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss

    @jax.jit
    def calculate_metrics(params, key, xs, ys, k=5):
        logits = transformer.apply(params, key, xs)
        classes = logits.argmax(axis=-1)
        accuracy = jnp.mean(classes == ys)

        top_k = jnp.argsort(logits, axis=-1)[:, -k:]
        hits = (ys == top_k.T).any(axis=0)
        top_k_accuracy = jnp.mean(hits)

        metrics = {
            "accuracy": accuracy,
            f"top_{k}_acc": top_k_accuracy,
        }
        return metrics

    @jax.jit
    def update(
        params: hk.Params,
        opt_state: optax.OptState,
        key: jax.random.PRNGKey,
        xs: tf.Tensor,
        ys: tf.Tensor
    ) -> t.Tuple[hk.Params, optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, key, xs, ys)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state, loss

    if not args.debug:
        mlflow.set_experiment("cifar_haiku")
        mlflow.start_run()
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("k", k)
        mlflow.log_param("heads", heads)
        mlflow.log_param("depth", depth)
        mlflow.log_param("patch_size", str(patch_size))
        mlflow.log_param("image_size", str(image_size))
        mlflow.log_param("dropout", dropout)

    for e in range(epochs):
        step = 0
        metrics_dict = defaultdict(lambda: 0)
        desc = f"Train Epoch {e}"
        train_bar = tqdm(train_ds, total=len(train_ds), ncols=0, desc=desc)

        for xs, ys in train_bar:
            key = next(rng_seq)
            params, opt_state, loss = update(params, opt_state, key, xs, ys)
            metrics = calculate_metrics(params, key, xs, ys)
            metrics["loss"] = loss

            step += 1
            metrics_dict = update_metrics(step, metrics_dict, metrics)
            if step % show_every == 0:
                metrics_display = {k: str(v)[:4] for k, v in metrics_dict.items()}
                train_bar.set_postfix(**metrics_display)

        train_metrics = {f"train_{k}": float(v) for k, v in metrics_dict.items()}
        if not args.debug:
            mlflow.log_metrics(train_metrics, step=e)

        step = 0
        metrics_dict = defaultdict(lambda: 0)
        desc = f"Valid Epoch {e}"
        val_bar = tqdm(val_ds, total=len(val_ds), ncols=0, desc=desc)

        for xs, ys in val_bar:
            key = next(rng_seq)
            loss = loss_fn(params, key, xs, ys)
            metrics = calculate_metrics(params, key, xs, ys)
            metrics["loss"] = loss

            step += 1
            metrics_dict = update_metrics(step, metrics_dict, metrics)
            if step % show_every == 0:
                metrics_display = {k: str(v)[:4] for k, v in metrics_dict.items()}
                val_bar.set_postfix(**metrics_display)

        val_metrics = {f"valid_{k}": float(v) for k, v in metrics_dict.items()}

        if not args.debug:
            mlflow.log_metrics(val_metrics, step=e)

        if e % save_every == 0 and not args.debug:
            pickle.dump(params, open("weights.pkl", "wb"))
            mlflow.log_artifact("weights.pkl", "weights")
            pickle.dump(opt_state, open("optimizer.pkl", "wb"))
            mlflow.log_artifact("optimizer.pkl", "optimizer")


if __name__ == "__main__":
    main()
