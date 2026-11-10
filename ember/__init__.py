"""Ember — a from-scratch autograd + neural-network library in pure Python.

Ember implements reverse-mode automatic differentiation and the building
blocks of a small deep-learning framework (layers, activations, losses,
optimizers, data loading) **with zero runtime dependencies**. It is a
teaching-grade engine written to be read: every operation's forward math and
backward rule lives in plain Python.

Typical usage::

    import ember as eb

    model = eb.Sequential(eb.nn.Linear(2, 8), eb.nn.Tanh(), eb.nn.Linear(8, 1))
    opt = eb.SGD(model.parameters(), lr=0.1)

    for x, y in loader:
        pred = model(x)
        loss = eb.nn.MSELoss()(pred, y)
        loss.backward()
        opt.step()
        model.zero_grad()
"""

from __future__ import annotations

from .tensor import Tensor
from . import ops
from . import _array as array
from .nn import (
    Module, Linear, ReLU, Sigmoid, Tanh, Softmax,
    MSELoss, BCEWithLogitsLoss, CrossEntropyLoss, Sequential,
    kaiming_uniform, xavier_uniform, zeros_init,
)
from .optim import SGD
from .data import TensorDataset, DataLoader

__version__ = "1.0.0"

__all__ = [
    "Tensor", "ops", "array",
    "Module", "Linear", "ReLU", "Sigmoid", "Tanh", "Softmax",
    "MSELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "Sequential",
    "kaiming_uniform", "xavier_uniform", "zeros_init",
    "SGD", "TensorDataset", "DataLoader",
]
