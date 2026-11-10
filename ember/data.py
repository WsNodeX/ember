"""Minimal dataset / data-loader utilities (no external deps)."""

from __future__ import annotations

import random as _random

from .tensor import Tensor

__all__ = ["TensorDataset", "DataLoader"]


class TensorDataset:
    """Packs aligned input/target samples into a dataset.

    Each sample is a tuple ``(x, y)`` where ``x`` and ``y`` may be
    :class:`Tensor` or Python scalars/lists.
    """

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("TensorDataset needs at least one tensor column")
        self.columns = [self._coerce(t) for t in tensors]
        self.length = len(self.columns[0])

    @staticmethod
    def _coerce(t):
        if isinstance(t, Tensor):
            return t
        if isinstance(t, list):
            return [Tensor(x) if not isinstance(x, Tensor) else x for x in t]
        raise TypeError("unsupported dataset column type")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(col[idx] for col in self.columns)


class DataLoader:
    """Iterates a dataset in mini-batches, optionally shuffling."""

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False,
                 rng: _random.Random = None):
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.rng = rng or _random.Random(0)

    def __iter__(self):
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, n, self.batch_size):
            chunk = indices[start:start + self.batch_size]
            batch = tuple(
                _stack([self.dataset[i][col] for i in chunk])
                for col in range(len(self.dataset.columns))
            )
            yield batch


def _stack(tensors):
    """Stack a list of scalar/vector Tensors into a batched Tensor."""
    if all(t.shape == () for t in tensors):
        return Tensor([t.data for t in tensors])  # (B,) vector
    if all(t.shape and len(t.shape) == 1 for t in tensors):
        return Tensor([list(t.data) for t in tensors])  # (B, F)
    # mixed / scalar+vector fallback: keep as list of tensors
    return tensors
