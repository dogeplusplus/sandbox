import jax
import numpy as np
import jax.numpy as jnp

from typing import NamedTuple, Dict, Callable


frame_stack = []

class Frame(NamedTuple):
    params: Dict[str, jnp.ndarray]
    is_initialising: bool = False


def current_frame():
    return frame_stack[-1]

class Transformed(NamedTuple):
    init: Callable
    apply: Callable


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


def get_param(identifier, shape):
    if current_frame().is_initialising:
        current_frame().params[identifier] = np.random.normal(size=shape)

    return current_frame().params[identifier]


def parameter_shapes(params):
    return jax.tree_map(lambda p: p.shape, params)


class Linear:
    def __init__(self, width):
        self._width = width

    def __call__(self, x):
        w = get_param("w", shape=(x.shape[-1], self._width))
        b = get_param("b", shape=(self._width,))
        return x @ w + b

init, apply = transform(Linear(4))

data = jnp.ones((2, 3))
params = init(data)
print(parameter_shapes(params))
print(apply(params, data))

init2, apply2 = transform(Linear(5))
params2 = init2(data)
print(parameter_shapes(params2))
print(apply(params2, data))

print(frame_stack)

import dataclasses
import collections

@dataclasses.dataclass
class Frame:
    params: Dict[str, jnp.ndarray]
    is_initialising: bool = False

    module_counts: Dict[str, int] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(lambda: 0))

    call_stack: list = dataclasses.field(default_factory=list)

    def create_param_path(self, identifier) -> str:
        return "/".join(["~"] + self.call_stack + [identifier])

    def create_unique_module_name(self, module_name: str) -> str:
        number = self.module_counts[module_name]
        self.module_counts[module_name] += 1
        return f"{module_name}_{number}"

frame_stack = []

