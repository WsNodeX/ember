"""Pure-Python dense array math for 0-D (scalar), 1-D and 2-D tensors.

This module intentionally avoids NumPy so that *ember* has **zero runtime
dependencies** and runs anywhere a CPython interpreter exists. All math is
expressed over nested ``list`` structures plus the scalar ``float`` type:

* scalar  -> a ``float`` with shape ``()``
* vector  -> ``List[float]`` with shape ``(n,)``
* matrix  -> ``List[List[float]]`` with shape ``(m, n)``

The functions here are deliberately small and composable; the autograd
engine in :mod:`ember.tensor` builds on top of them.
"""

from __future__ import annotations

import math
import operator
import random as _random

__all__ = [
    "is_scalar",
    "shape_of",
    "numel",
    "zeros",
    "ones",
    "full",
    "randn",
    "ew",
    "broadcast_ew",
    "sum_all",
    "sum_axis",
    "mean_axis",
    "matmul",
    "transpose",
    "exp",
    "log",
    "sqrt",
    "neg",
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
]


def is_scalar(x) -> bool:
    """Return ``True`` when *x* is a Python scalar (int/float)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def shape_of(data):
    """Infer the shape tuple of a nested-list / scalar datum."""
    if is_scalar(data):
        return ()
    if isinstance(data, list):
        if not data:
            return (0,)
        shape = [len(data)]
        inner = data[0]
        while isinstance(inner, list):
            shape.append(len(inner))
            inner = inner[0] if inner else None
        return tuple(shape)
    raise TypeError(f"unsupported data type: {type(data)!r}")


def numel(shape: tuple) -> int:
    """Number of elements described by *shape*."""
    n = 1
    for d in shape:
        n *= d
    return n


def zeros(shape: tuple):
    """All-zero datum matching *shape*."""
    if shape == ():
        return 0.0
    if len(shape) == 1:
        return [0.0] * shape[0]
    if len(shape) == 2:
        return [[0.0] * shape[1] for _ in range(shape[0])]
    raise ValueError(f"unsupported shape {shape}")


def ones(shape: tuple):
    """All-one datum matching *shape*."""
    if shape == ():
        return 1.0
    if len(shape) == 1:
        return [1.0] * shape[0]
    if len(shape) == 2:
        return [[1.0] * shape[1] for _ in range(shape[0])]
    raise ValueError(f"unsupported shape {shape}")


def full(shape: tuple, value: float):
    """Constant datum matching *shape* filled with *value*."""
    if shape == ():
        return float(value)
    if len(shape) == 1:
        return [float(value)] * shape[0]
    if len(shape) == 2:
        return [[float(value)] * shape[1] for _ in range(shape[0])]
    raise ValueError(f"unsupported shape {shape}")


def _gaussian(rng: _random.Random) -> float:
    """Box-Muller transform returning a standard-normal sample."""
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z


def randn(shape: tuple, rng: _random.Random):
    """Standard-normal random datum with the given *shape* using *rng*."""
    if shape == ():
        return _gaussian(rng)
    if len(shape) == 1:
        return [_gaussian(rng) for _ in range(shape[0])]
    if len(shape) == 2:
        return [[_gaussian(rng) for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError(f"unsupported shape {shape}")


def _ew_scalar_bcast(op, x, sx, sval):
    """Apply ``op`` broadcasting scalar *sval* over datum *x* of shape *sx*."""
    if sx == ():
        return op(x, sval)
    if len(sx) == 1:
        return [op(v, sval) for v in x]
    if len(sx) == 2:
        return [[op(v, sval) for v in row] for row in x]
    raise ValueError(f"unsupported shape {sx}")


def ew(op, a, b):
    """Elementwise ``op`` over two equally shaped nested-list data."""
    if is_scalar(a) and is_scalar(b):
        return op(a, b)
    if is_scalar(a):
        return _ew_scalar_bcast(op, b, shape_of(b), a)
    if is_scalar(b):
        return _ew_scalar_bcast(op, a, shape_of(a), b)
    return [ew(op, x, y) for x, y in zip(a, b)]


def broadcast_ew(op, a, sa, b, sb):
    """Elementwise ``op`` with scalar and ``(m, n)`` x ``(n,)`` broadcasting."""
    if sa == ():
        return _ew_scalar_bcast(op, b, sb, a)
    if sb == ():
        return _ew_scalar_bcast(op, a, sa, b)
    if sa == sb:
        return ew(op, a, b)
    # 2-D datum with a 1-D datum broadcast along the last axis.
    if len(sa) == 2 and len(sb) == 1 and sa[1] == sb[0]:
        m, n = sa
        return [[op(a[r][c], b[c]) for c in range(n)] for r in range(m)]
    if len(sb) == 2 and len(sa) == 1 and sb[1] == sa[0]:
        m, n = sb
        return [[op(a[c], b[r][c]) for c in range(n)] for r in range(m)]
    raise ValueError(f"cannot broadcast shapes {sa} and {sb}")


def _scale(x, s):
    """Multiply every element of *x* by scalar *s* (any shape)."""
    if is_scalar(x):
        return x * s
    if isinstance(x, list):
        return [_scale(v, s) for v in x]
    return x * s


def sum_all(a) -> float:
    """Sum every element of *a* into a single scalar."""
    if is_scalar(a):
        return a
    if isinstance(a, list):
        return sum(sum_all(v) for v in a)
    return a


def sum_axis(a, shape, axis):
    """Sum *a* over *axis*.

    * scalar -> itself
    * 1-D    -> scalar (axis ignored)
    * 2-D    -> ``axis in (0, -2)`` collapses rows -> ``(n,)``;
                ``axis in (1, -1)`` collapses columns -> ``(m,)``
    """
    if shape == ():
        return a
    if len(shape) == 1:
        return sum_all(a)
    m, n = shape
    if axis in (0, -2):
        return [sum(a[r][c] for r in range(m)) for c in range(n)]
    return [sum(a[r]) for r in range(m)]


def mean_axis(a, shape, axis):
    """Mean of *a* over *axis*."""
    if shape == ():
        return a
    if len(shape) == 1:
        return sum_all(a) / max(1, shape[0])
    m, n = shape
    count = m if axis in (1, -1) else n
    s = sum_axis(a, shape, axis)
    return _scale(s, 1.0 / max(1, count))


def matmul(a, sa, b, sb):
    """Matrix / vector product returning ``(data, shape)``.

    Supported pairings:
        (m, k) @ (k, n) -> (m, n)
        (m, k) @ (k,)   -> (m,)
        (k,)   @ (k, n) -> (n,)
        (k,)   @ (k,)   -> scalar
        scalar @ tensor -> scaled tensor
        tensor @ scalar -> scaled tensor
    """
    if is_scalar(a) or is_scalar(b):
        s, t = (a, b) if is_scalar(a) else (b, a)
        return _scale(t, s), (sa if is_scalar(b) else sb)

    if len(sa) == 1 and len(sb) == 1:
        return sum(x * y for x, y in zip(a, b)), ()

    if len(sa) == 2 and len(sb) == 1:
        m, k = sa
        return [sum(a[r][c] * b[c] for c in range(k)) for r in range(m)], (m,)

    if len(sa) == 1 and len(sb) == 2:
        k, n = sb
        return [sum(a[c] * b[c][cc] for c in range(k)) for cc in range(n)], (n,)

    if len(sa) == 2 and len(sb) == 2:
        m, k = sa
        _, n = sb
        return [
            [sum(a[r][c] * b[c][cc] for c in range(k)) for cc in range(n)]
            for r in range(m)
        ], (m, n)

    raise ValueError(f"matmul incompatible shapes {sa}, {sb}")


def transpose(a, sa):
    """Transpose. 2-D -> swapped; 1-D / scalar -> unchanged."""
    if sa == () or len(sa) == 1:
        return a, sa
    if len(sa) == 2:
        m, n = sa
        return [[a[r][c] for r in range(m)] for c in range(n)], (n, m)
    raise ValueError(f"transpose unsupported for shape {sa}")


def exp(a, shape):
    if shape == ():
        return math.exp(a)
    if len(shape) == 1:
        return [math.exp(v) for v in a]
    return [[math.exp(v) for v in row] for row in a]


def log(a, shape):
    if shape == ():
        return math.log(a)
    if len(shape) == 1:
        return [math.log(v) for v in a]
    return [[math.log(v) for v in row] for row in a]


def sqrt(a, shape):
    if shape == ():
        return math.sqrt(a)
    if len(shape) == 1:
        return [math.sqrt(v) for v in a]
    return [[math.sqrt(v) for v in row] for row in a]


def neg(a, shape):
    return _scale(a, -1.0)


def relu(a, shape):
    if shape == ():
        return a if a > 0.0 else 0.0
    if len(shape) == 1:
        return [v if v > 0.0 else 0.0 for v in a]
    return [[v if v > 0.0 else 0.0 for v in row] for row in a]


def sigmoid(a, shape):
    if shape == ():
        return 1.0 / (1.0 + math.exp(-a))
    if len(shape) == 1:
        return [1.0 / (1.0 + math.exp(-v)) for v in a]
    return [[1.0 / (1.0 + math.exp(-v)) for v in row] for row in a]


def tanh(a, shape):
    if shape == ():
        return math.tanh(a)
    if len(shape) == 1:
        return [math.tanh(v) for v in a]
    return [[math.tanh(v) for v in row] for row in a]


def softmax(a, shape, axis=-1):
    """Numerically stable softmax (last axis for matrices)."""
    if shape == ():
        return 1.0
    if len(shape) == 1:
        mx = max(a)
        e = [math.exp(v - mx) for v in a]
        s = sum(e) or 1.0
        return [v / s for v in e]
    if len(shape) == 2:
        out = []
        for row in a:
            mx = max(row)
            e = [math.exp(v - mx) for v in row]
            s = sum(e) or 1.0
            out.append([v / s for v in e])
        return out
    raise ValueError(f"softmax unsupported for shape {shape}")
