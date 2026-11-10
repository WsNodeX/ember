"""Activation functions packaged as parameter-free modules."""

from __future__ import annotations

from ..tensor import Tensor
from .module import Module

__all__ = ["ReLU", "Sigmoid", "Tanh", "Softmax"]


class ReLU(Module):
    """Rectified linear unit ``max(0, x)``."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    """Logistic sigmoid ``1 / (1 + exp(-x))``."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    """Hyperbolic tangent."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Softmax(Module):
    """Softmax over the last axis (rows for matrices)."""

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)
