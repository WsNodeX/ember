"""Parameter initializers (deterministic via a seeded RNG)."""

from __future__ import annotations

import math
import random as _random

from .. import _array as A
from ..tensor import Tensor

__all__ = ["kaiming_uniform", "xavier_uniform", "zeros_init"]


def kaiming_uniform(shape: tuple, gain: float = 1.0, rng: _random.Random = None):
    """He / Kaiming uniform initializer (good with ReLU networks)."""
    rng = rng or _random.Random(0)
    if len(shape) == 1:
        fan_in = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]  # weight stored as (in_features, out_features)
    else:
        fan_in = 1
    bound = gain * math.sqrt(3.0 / max(1, fan_in))
    if shape == ():
        return Tensor((rng.random() * 2 - 1) * bound)
    if len(shape) == 1:
        return Tensor([(rng.random() * 2 - 1) * bound for _ in range(shape[0])])
    return Tensor(
        [[(rng.random() * 2 - 1) * bound for _ in range(shape[1])]
         for _ in range(shape[0])]
    )


def xavier_uniform(shape: tuple, rng: _random.Random = None):
    """Glorot / Xavier uniform initializer."""
    rng = rng or _random.Random(0)
    if len(shape) == 2:
        fan_in, fan_out = shape[0], shape[1]  # (in_features, out_features)
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    else:
        fan_in = fan_out = 1
    bound = math.sqrt(6.0 / max(1, fan_in + fan_out))
    if shape == ():
        return Tensor((rng.random() * 2 - 1) * bound)
    if len(shape) == 1:
        return Tensor([(rng.random() * 2 - 1) * bound for _ in range(shape[0])])
    return Tensor(
        [[(rng.random() * 2 - 1) * bound for _ in range(shape[1])]
         for _ in range(shape[0])]
    )


def zeros_init(shape: tuple):
    return Tensor(A.zeros(shape))
