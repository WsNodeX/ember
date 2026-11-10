"""Sequential container and convenience re-exports."""

from __future__ import annotations

from .module import Module

__all__ = ["Sequential"]


class Sequential(Module):
    """Chains modules (and callables) in order."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def extra_repr(self) -> str:
        return f"{len(self.layers)} layers"
