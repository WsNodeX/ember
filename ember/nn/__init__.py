"""Neural-network layers, activations and losses for *ember*."""

from __future__ import annotations

from .module import Module
from .init import kaiming_uniform, xavier_uniform, zeros_init
from .linear import Linear
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .loss import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from .sequential import Sequential

__all__ = [
    "Module", "Linear", "ReLU", "Sigmoid", "Tanh", "Softmax",
    "MSELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "Sequential",
    "kaiming_uniform", "xavier_uniform", "zeros_init",
]
