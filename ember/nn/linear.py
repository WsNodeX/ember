"""Fully-connected (dense) layer."""

from __future__ import annotations

import random as _random

from ..tensor import Tensor
from .init import kaiming_uniform, zeros_init
from .module import Module

__all__ = ["Linear"]


class Linear(Module):
    """Applies an affine transform ``y = x W^T + b``.

    Parameters
    ----------
    in_features, out_features:
        Sizes of the input and output vectors.
    bias:
        When ``True`` (default) a learnable additive bias is included.
    rng:
        Optional seeded RNG for reproducible initialization.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rng: _random.Random = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = kaiming_uniform((in_features, out_features), rng=rng)
        self.weight.requires_grad = True
        if bias:
            self.bias = zeros_init((out_features,))
            self.bias.requires_grad = True
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, in_features) -> (batch, out_features)
        from .. import ops
        out = ops.matmul(x, self.weight)
        if self.bias is not None:
            out = ops.add(out, self.bias)
        return out

    def extra_repr(self) -> str:
        return f"{self.in_features} -> {self.out_features}"
