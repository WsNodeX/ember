"""Neural-network building blocks for *ember*."""

from __future__ import annotations

from ..tensor import Tensor

__all__ = ["Module"]


class Module:
    """Base class for everything that holds trainable :class:`Tensor` weights.

    Subclasses implement :meth:`forward`. Calling the module invokes
    ``forward``. :meth:`parameters` walks the object graph (including nested
    modules, lists, tuples and dicts) to collect every leaf tensor that
    requires a gradient.
    """

    def parameters(self):
        out = []
        seen = set()

        def rec(obj):
            if isinstance(obj, Tensor):
                if id(obj) not in seen:
                    seen.add(id(obj))
                    out.append(obj)
            elif isinstance(obj, Module):
                for v in obj.__dict__.values():
                    rec(v)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    rec(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    rec(item)

        rec(self)
        return out

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
