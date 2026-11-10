"""Loss functions with analytic, numerically stable backward passes."""

from __future__ import annotations

import math

from .. import _array as A
from ..tensor import Tensor
from .module import Module

__all__ = ["MSELoss", "BCEWithLogitsLoss", "CrossEntropyLoss"]


class MSELoss(Module):
    """Mean squared error ``mean((pred - target)^2)``."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        from .. import ops
        diff = ops.sub(pred, target)
        sq = ops.mul(diff, diff)
        return ops.mean(sq)


class BCEWithLogitsLoss(Module):
    """Binary cross entropy with a fused, stable sigmoid.

    Expects raw logits (not probabilities) and 0/1 targets.
    """

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        s = A.sigmoid(logits.data, logits.shape)
        n = A.numel(logits.shape)
        total = 0.0
        if logits.shape == ():
            total = -(targets.data * math.log(s) +
                      (1 - targets.data) * math.log(1 - s))
        elif len(logits.shape) == 1:
            for i in range(len(s)):
                total += -(targets.data[i] * math.log(s[i]) +
                          (1 - targets.data[i]) * math.log(1 - s[i]))
        else:
            for r in range(len(s)):
                for c in range(len(s[r])):
                    total += -(targets.data[r][c] * math.log(s[r][c]) +
                              (1 - targets.data[r][c]) * math.log(1 - s[r][c]))
        loss = Tensor(total / max(1, n), requires_grad=logits.requires_grad)
        loss._prev = {logits, targets}

        def _backward():
            g = loss.grad
            # dL/dz = (s - t) / N  (per element), scaled by loss.grad
            if logits.shape == ():
                local = (s - targets.data) / max(1, n)
                logits._accumulate(g * local)
            elif len(logits.shape) == 1:
                local = [(s[i] - targets.data[i]) / max(1, n)
                         for i in range(len(s))]
                logits._accumulate(_scale_list(local, g))
            else:
                local = [[(s[r][c] - targets.data[r][c]) / max(1, n)
                          for c in range(len(s[r]))] for r in range(len(s))]
                logits._accumulate(_scale_list2(local, g))

        loss._backward = _backward
        return loss


class CrossEntropyLoss(Module):
    """Softmax cross entropy over class logits.

    ``targets`` is a list of integer class indices (length = batch size).
    """

    def forward(self, logits: Tensor, targets) -> Tensor:
        if logits.shape == ():
            raise ValueError("CrossEntropyLoss expects batched logits (N, C)")
        n, c = logits.shape
        if isinstance(targets, Tensor):
            td = targets.data
            flat = td if A.is_scalar(td) else [v for row in td for v in
                                              (row if isinstance(row, list) else [row])]
            targets = [int(round(v)) for v in flat]
        probs = A.softmax(logits.data, (n, c))
        total = 0.0
        for i in range(n):
            total += -math.log(max(probs[i][targets[i]], 1e-12))
        loss = Tensor(total / n, requires_grad=logits.requires_grad)
        loss._prev = {logits}

        def _backward():
            g = loss.grad
            # dL/dz_i = (p_i - 1{target}) / N
            grad = [[(probs[i][j] - (1.0 if j == targets[i] else 0.0)) / n
                     for j in range(c)] for i in range(n)]
            if g != 1.0:
                grad = [[v * g for v in row] for row in grad]
            logits._accumulate(grad)

        loss._backward = _backward
        return loss


def _scale_list(data, s):
    return [v * s for v in data]


def _scale_list2(data, s):
    return [[v * s for v in row] for row in data]
