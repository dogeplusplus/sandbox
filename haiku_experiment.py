import jax
import typing as t
import dataclasses
import collections
import numpy as np
import jax.numpy as jnp


frame_stack = []

class Transformed(t.NamedTuple):
  init: t.Callable
  apply: t.Callable


def transform(f) -> Transformed:

  def init_f(*args, **kwargs):
    frame_stack.append(Frame({}, is_initialising=True))
    f(*args, **kwargs)
    frame = frame_stack.pop()
    return frame.params

  def apply_f(params, *args, **kwargs):
    frame_stack.append(Frame(params))
    outs = f(*args, **kwargs)
    frame_stack.pop()
    return outs

  return Transformed(init_f, apply_f)


@dataclasses.dataclass
class Frame:
    params: t.Dict[str, jnp.ndarray]
    is_initialising: bool = False

    module_counts: t.Dict[str, int] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(lambda: 0))

    call_stack: list = dataclasses.field(default_factory=list)

    def create_param_path(self, identifier) -> str:
        return "/".join(["~"] + self.call_stack + [identifier])

    def create_unique_module_name(self, module_name: str) -> str:
        number = self.module_counts[module_name]
        self.module_counts[module_name] += 1
        return f"{module_name}_{number}"


def current_frame():
    return frame_stack[-1]


class Module:
    def __init__(self):
        self._unique_name = current_frame().create_unique_module_name(self.__class__.__name__)


def module_method(f):
    def wrapped(self, *args, **kwargs):
        module_name = self._unique_name
        call_stack = current_frame().call_stack
        call_stack.append(module_name)
        call_stack.append(f.__name__)
        outs = f(self, *args, **kwargs)
        assert call_stack.pop() == f.__name__
        assert call_stack.pop() == module_name

        return outs

    return wrapped


def get_param(identifier, shape):
    frame = current_frame()
    param_path = frame.create_param_path(identifier)

    if frame.is_initialising:
        frame.params[param_path] = np.random.normal(size=shape)

    return frame.params[param_path]


def parameter_shapes(params):
  return jax.tree_map(lambda p: p.shape, params)


class Linear(Module):
    def __init__(self, width):
        super().__init__()
        self._width = width

    @module_method
    def __call__(self, x):
        w = get_param("w", shape=(x.shape[-1], self._width))
        b = get_param("b", shape=(self._width,))
        return x @ w + b

class MLP(Module):
    def __init__(self, widths):
        super().__init__()
        self._widths = widths

    @module_method
    def __call__(self, x):
        for w in self._widths:
            out = Linear(w)(x)
            x = jax.nn.sigmoid(out)

        return out

class ParameterReuseTest(Module):
    @module_method
    def __call__(self, x):
        f = Linear(x.shape[-1])

        x = f(x)
        x = jax.nn.relu(x)
        return f(x)


if __name__ == "__main__":
    data = jnp.ones((2, 3))
    init, apply = transform(lambda x: MLP([3,5])(x))

    params = init(data)
    print(parameter_shapes(params))
    print(apply(params, data))

    init, forward = transform(lambda x: ParameterReuseTest()(x))
    print(parameter_shapes(init(data)))

    import matplotlib.pyplot as plt

    xs = np.linspace(-2., 2., num=128)[:,None]
    ys = xs ** 2

    def mlp(x):
        return MLP([128, 128, 1])(x)

    init, forward = transform(mlp)
    params = init(xs)
    parameter_shapes(params)

    def loss_fn(params, x, y):
        return jnp.mean((forward(params, x) - y) ** 2)

    LR = 1e-3

    @jax.jit
    def update(params, x, y):
        grads = jax.grad(loss_fn)(params, x, y)
        return jax.tree_multimap(
            lambda p, g: p - LR * g, params, grads
        )

    for _ in range(5000):
        params = update(params, xs, ys)

    plt.scatter(xs, ys, label="data")
    plt.scatter(xs, forward(params, xs), label="prediction")
    plt.legend()
    plt.show()

