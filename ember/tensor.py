"""Autograd :class:`Tensor` with a reverse-mode (backprop) engine.

The design mirrors the mental model used by modern deep-learning frameworks:
a forward pass builds a dynamic directed graph of operations, and calling
:meth:`Tensor.backward` walks that graph in reverse topological order,
accumulating gradients. Because the math lives in :mod:`ember._array`, this
module only has to manage the graph bookkeeping.
"""

from __future__ import annotations

import operator

from . import _array as A

__all__ = ["Tensor"]


class Tensor:
    """A differentiable scalar / vector / matrix.

    Parameters
    ----------
    data:
        Nested ``list`` (vector / matrix) or ``float`` (scalar).
    requires_grad:
        When ``True`` gradients are accumulated for this tensor.
    """

    __slots__ = ("data", "shape", "requires_grad", "grad", "_backward", "_prev")

    def __init__(self, data, requires_grad: bool = False):
        if not (A.is_scalar(data) or isinstance(data, list)):
            # Accept floats that numpy would treat as scalars; coerce ints.
            if isinstance(data, (int, float)) and not isinstance(data, bool):
                data = float(data)
            else:
                raise TypeError("data must be float, list[float] or list[list[float]]")
        self.data = data
        self.shape = A.shape_of(data)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    # -- introspection -------------------------------------------------
    def item(self) -> float:
        if self.shape != ():
            raise ValueError("item() only works on a scalar tensor")
        return float(self.data)

    def tolist(self):
        return self.data

    def __repr__(self) -> str:
        return f"Tensor({self.data!r}, shape={self.shape}, requires_grad={self.requires_grad})"

    # -- mutation helpers ---------------------------------------------
    def _accumulate(self, g) -> None:
        """Add gradient *g* (same shape) into ``self.grad``."""
        if not self.requires_grad or g is None:
            return
        if self.grad is None:
            self.grad = A.zeros(self.shape)
        self.grad = A.ew(operator.add, self.grad, g)

    def zero_grad(self) -> None:
        self.grad = None

    # -- autograd ------------------------------------------------------
    def backward(self) -> None:
        """Run reverse-mode automatic differentiation from this tensor."""
        topo: list["Tensor"] = []
        visited = set()

        def build(v: "Tensor") -> None:
            vid = id(v)
            if vid in visited:
                return
            visited.add(vid)
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)

        for v in topo:
            if v.grad is None:
                v.grad = A.zeros(v.shape)

        # Seed: gradient of the scalar output w.r.t. itself is one.
        if self.shape == ():
            self.grad = 1.0
        else:
            self.grad = A.ones(self.shape)

        for v in reversed(topo):
            v._backward()

    # -- operators (delegated to ember.ops) ----------------------------
    def __add__(self, other):
        from . import ops
        return ops.add(self, other)

    def __radd__(self, other):
        from . import ops
        return ops.add(other, self)

    def __sub__(self, other):
        from . import ops
        return ops.sub(self, other)

    def __rsub__(self, other):
        from . import ops
        return ops.sub(other, self)

    def __mul__(self, other):
        from . import ops
        return ops.mul(self, other)

    def __rmul__(self, other):
        from . import ops
        return ops.mul(other, self)

    def __truediv__(self, other):
        from . import ops
        return ops.div(self, other)

    def __rtruediv__(self, other):
        from . import ops
        return ops.div(other, self)

    def __neg__(self):
        from . import ops
        return ops.neg(self)

    def matmul(self, other):
        from . import ops
        return ops.matmul(self, other)

    def __matmul__(self, other):
        from . import ops
        return self.matmul(other)

    def sum(self):
        from . import ops
        return ops.sum(self)

    def mean(self):
        from . import ops
        return ops.mean(self)

    def transpose(self):
        from . import ops
        return ops.transpose(self)

    def relu(self):
        from . import ops
        return ops.relu(self)

    def sigmoid(self):
        from . import ops
        return ops.sigmoid(self)

    def tanh(self):
        from . import ops
        return ops.tanh(self)

    def softmax(self, axis=-1):
        from . import ops
        return ops.softmax(self, axis=axis)

    def exp(self):
        from . import ops
        return ops.exp(self)

    def log(self):
        from . import ops
        return ops.log(self)

    def sqrt(self):
        from . import ops
        return ops.sqrt(self)

    def pow(self, exponent: float):
        from . import ops
        return ops.pow(self, exponent)

    def neg(self):
        from . import ops
        return ops.neg(self)
