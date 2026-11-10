"""Stochastic-gradient-descent optimizer with momentum and weight decay."""

from __future__ import annotations

import operator

from . import _array as A
from .tensor import Tensor

__all__ = ["SGD"]


class SGD:
    """Classical SGD with optional momentum and L2 weight decay.

    Parameters
    ----------
    params:
        Iterable of :class:`Tensor` parameters (e.g. ``model.parameters()``).
    lr:
        Learning rate.
    momentum:
        Momentum factor (0 disables momentum).
    weight_decay:
        L2 penalty coefficient (0 disables).
    nesterov:
        When ``True`` applies Nesterov-accelerated gradient.
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._buf = {}

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad

            if self.weight_decay:
                if self.weight_decay != 0.0:
                    g = A.ew(operator.add, g, A.ew(
                        lambda x, y: x * y, p.data, A.full(p.shape, self.weight_decay)))

            if self.momentum:
                prev = self._buf.get(id(p), A.zeros(p.shape))
                if self.nesterov:
                    # v = m*v - lr*g ; then x += m*v - lr*g  (Nesterov look-ahead)
                    v = A.ew(operator.add,
                             A.ew(operator.mul, prev, self.momentum),
                             A.ew(operator.mul, g, -self.lr))
                    self._buf[id(p)] = v
                    update = A.ew(operator.add, A.ew(operator.mul, v, self.momentum),
                                  A.ew(operator.mul, g, -self.lr))
                else:
                    v = A.ew(operator.add,
                             A.ew(operator.mul, prev, self.momentum),
                             A.ew(operator.mul, g, -self.lr))
                    self._buf[id(p)] = v
                    update = v
            else:
                update = A.ew(operator.mul, g, -self.lr)

            p.data = A.ew(operator.add, p.data, update)
            p.grad = None

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
