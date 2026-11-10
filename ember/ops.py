"""Differentiable operations built on :mod:`ember._array`.

Every function here returns a fresh :class:`~ember.tensor.Tensor` and wires up
its ``_backward`` closure so that :meth:`Tensor.backward` can propagate
gradients back through the dynamic graph.
"""

from __future__ import annotations

import math
import operator
import builtins

from . import _array as A
from .tensor import Tensor

__all__ = [
    "add", "sub", "mul", "div", "neg", "matmul", "transpose",
    "sum", "mean", "pow", "exp", "log", "sqrt", "relu", "sigmoid",
    "tanh", "softmax",
]


def _wrap(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(float(x))


def _reduce_grad(g, g_shape, target_shape):
    """Reduce a full-shape gradient *g* to the shape of one operand."""
    if target_shape == ():
        return A.sum_all(g)
    if target_shape == g_shape:
        return g
    # Operand was a 1-D vector broadcast across a 2-D output.
    return A.sum_axis(g, g_shape, axis=0)


# --------------------------------------------------------------------------- #
# Elementwise + broadcast                                                      #
# --------------------------------------------------------------------------- #
def add(a, b):
    a, b = _wrap(a), _wrap(b)
    data = A.broadcast_ew(operator.add, a.data, a.shape, b.data, b.shape)
    out = Tensor(data, requires_grad=a.requires_grad or b.requires_grad)
    out._prev = {a, b}

    def _backward():
        g = out.grad
        ga = _reduce_grad(g, out.shape, a.shape)
        gb = _reduce_grad(g, out.shape, b.shape)
        a._accumulate(ga)
        b._accumulate(gb)

    out._backward = _backward
    return out


def sub(a, b):
    a, b = _wrap(a), _wrap(b)
    data = A.broadcast_ew(operator.sub, a.data, a.shape, b.data, b.shape)
    out = Tensor(data, requires_grad=a.requires_grad or b.requires_grad)
    out._prev = {a, b}

    def _backward():
        g = out.grad
        ga = _reduce_grad(g, out.shape, a.shape)
        gb = _reduce_grad(A.neg(g, out.shape), out.shape, b.shape)
        a._accumulate(ga)
        b._accumulate(gb)

    out._backward = _backward
    return out


def mul(a, b):
    a, b = _wrap(a), _wrap(b)
    data = A.broadcast_ew(operator.mul, a.data, a.shape, b.data, b.shape)
    out = Tensor(data, requires_grad=a.requires_grad or b.requires_grad)
    out._prev = {a, b}

    def _backward():
        g = out.grad
        ga_full = A.broadcast_ew(operator.mul, g, out.shape, b.data, b.shape)
        gb_full = A.broadcast_ew(operator.mul, g, out.shape, a.data, a.shape)
        a._accumulate(_reduce_grad(ga_full, out.shape, a.shape))
        b._accumulate(_reduce_grad(gb_full, out.shape, b.shape))

    out._backward = _backward
    return out


def div(a, b):
    a, b = _wrap(a), _wrap(b)
    data = A.broadcast_ew(operator.truediv, a.data, a.shape, b.data, b.shape)
    out = Tensor(data, requires_grad=a.requires_grad or b.requires_grad)
    out._prev = {a, b}

    def _backward():
        g = out.grad
        # d/da = g / b ; d/db = -g * a / b^2
        recip_b = _recip(b.data, b.shape)
        ga_full = A.broadcast_ew(operator.mul, g, out.shape, recip_b, b.shape)
        a_over_b2 = A.broadcast_ew(
            lambda x, y: x / (y * y), a.data, a.shape, b.data, b.shape
        )
        gb_full = A.broadcast_ew(operator.mul, A.neg(g, out.shape), out.shape, a_over_b2, a.shape)
        a._accumulate(_reduce_grad(ga_full, out.shape, a.shape))
        b._accumulate(_reduce_grad(gb_full, out.shape, b.shape))

    out._backward = _backward
    return out


def _recip(data, shape):
    if shape == ():
        return 1.0 / data
    if len(shape) == 1:
        return [1.0 / v for v in data]
    return [[1.0 / v for v in row] for row in data]


def neg(a):
    a = _wrap(a)
    out = Tensor(A.neg(a.data, a.shape), requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        a._accumulate(_reduce_grad(A.neg(out.grad, out.shape), out.shape, a.shape))

    out._backward = _backward
    return out


def pow(a, exponent: float):
    a = _wrap(a)
    if a.shape == ():
        data = math.pow(a.data, exponent)
    else:
        data = _pow_data(a.data, a.shape, exponent)
    out = Tensor(data, requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        # d/dx x^p = p * x^(p-1)
        if a.shape == ():
            local = exponent * math.pow(a.data, exponent - 1.0)
        else:
            local = _scale_pow(a.data, a.shape, exponent)
        ga_full = A.broadcast_ew(operator.mul, out.grad, out.shape, local, a.shape)
        a._accumulate(_reduce_grad(ga_full, out.shape, a.shape))

    out._backward = _backward
    return out


def _scale_pow(data, shape, exponent):
    base = _pow_data(data, shape, exponent - 1.0)
    if shape == ():
        return exponent * base
    if len(shape) == 1:
        return [exponent * v for v in base]
    return [[exponent * v for v in row] for row in base]


def _pow_data(data, shape, p):
    if shape == ():
        return math.pow(data, p)
    if len(shape) == 1:
        return [math.pow(v, p) for v in data]
    return [[math.pow(v, p) for v in row] for row in data]


# --------------------------------------------------------------------------- #
# Matrix / vector                                                              #
# --------------------------------------------------------------------------- #
def matmul(a, b):
    a, b = _wrap(a), _wrap(b)
    data, shape = A.matmul(a.data, a.shape, b.data, b.shape)
    out = Tensor(data, requires_grad=a.requires_grad or b.requires_grad)
    out._prev = {a, b}

    def _backward():
        g = out.grad
        sa, sb = a.shape, b.shape
        if a.shape == () or b.shape == ():
            s, other, s_shape = (a, b, sa) if a.shape == () else (b, a, sb)
            if s_shape == ():  # 'a' is the scalar
                da_full = A.sum_all(A.ew(operator.mul, g, b.data))
                db_full = A.ew(operator.mul, g, a.data)
            else:  # 'b' is the scalar
                da_full = A.ew(operator.mul, g, b.data)
                db_full = A.sum_all(A.ew(operator.mul, g, a.data))
        elif len(sa) == 1 and len(sb) == 1:  # dot -> scalar
            da_full = A.ew(operator.mul, g, b.data)
            db_full = A.ew(operator.mul, g, a.data)
        elif len(sa) == 2 and len(sb) == 1:  # (m,k) @ (k,) -> (m,)
            m, k = sa
            ga = g
            da_full = [[ga[r] * b.data[c] for c in range(k)] for r in range(m)]
            db_full = [builtins.sum(a.data[r][c] * ga[r] for r in range(m)) for c in range(k)]
        elif len(sa) == 1 and len(sb) == 2:  # (k,) @ (k,n) -> (n,)
            k, n = sb
            gc = g
            da_full = [builtins.sum(gc[cc] * b.data[c][cc] for cc in range(n)) for c in range(k)]
            db_full = [[a.data[c] * gc[cc] for cc in range(n)] for c in range(k)]
        else:  # (m,k) @ (k,n)
            m, k = sa
            n = sb[1]
            bt, _ = A.transpose(b.data, sb)
            da_full = A.matmul(g, out.shape, bt, (n, k))[0]
            at, _ = A.transpose(a.data, sa)
            db_full = A.matmul(at, (k, m), g, out.shape)[0]
        a._accumulate(_reduce_grad(da_full, sa, a.shape))
        b._accumulate(_reduce_grad(db_full, sb, b.shape))

    out._backward = _backward
    return out


def transpose(a):
    a = _wrap(a)
    data, shape = A.transpose(a.data, a.shape)
    out = Tensor(data, requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        # transpose is its own inverse
        gd, _ = A.transpose(out.grad, out.shape)
        a._accumulate(gd)

    out._backward = _backward
    return out


# --------------------------------------------------------------------------- #
# Reductions                                                                   #
# --------------------------------------------------------------------------- #
def sum(a):
    a = _wrap(a)
    out = Tensor(A.sum_all(a.data), requires_grad=a.requires_grad)
    out._prev = {a}
    n = A.numel(a.shape)

    def _backward():
        a._accumulate(A.full(a.shape, out.grad))

    out._backward = _backward
    return out


def mean(a):
    a = _wrap(a)
    n = A.numel(a.shape)
    out = Tensor(A.sum_all(a.data) / max(1, n), requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        a._accumulate(A.full(a.shape, out.grad / max(1, n)))

    out._backward = _backward
    return out


# --------------------------------------------------------------------------- #
# Activations                                                                  #
# --------------------------------------------------------------------------- #
def relu(a):
    a = _wrap(a)
    out = Tensor(A.relu(a.data, a.shape), requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        local = _mask(a.data, a.shape)
        ga_full = A.broadcast_ew(operator.mul, out.grad, out.shape, local, a.shape)
        a._accumulate(_reduce_grad(ga_full, out.shape, a.shape))

    out._backward = _backward
    return out


def _mask(data, shape):
    if shape == ():
        return 1.0 if data > 0.0 else 0.0
    if len(shape) == 1:
        return [1.0 if v > 0.0 else 0.0 for v in data]
    return [[1.0 if v > 0.0 else 0.0 for v in row] for row in data]


def sigmoid(a):
    a = _wrap(a)
    s = A.sigmoid(a.data, a.shape)
    out = Tensor(s, requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        local = A.ew(lambda v, _u: v * (1.0 - v), s, s)
        ga_full = A.broadcast_ew(operator.mul, out.grad, out.shape, local, a.shape)
        a._accumulate(_reduce_grad(ga_full, out.shape, a.shape))

    out._backward = _backward
    return out


def tanh(a):
    a = _wrap(a)
    t = A.tanh(a.data, a.shape)
    out = Tensor(t, requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        local = A.ew(lambda v, _u: 1.0 - v * v, t, t)
        ga_full = A.broadcast_ew(operator.mul, out.grad, out.shape, local, a.shape)
        a._accumulate(_reduce_grad(ga_full, out.shape, a.shape))

    out._backward = _backward
    return out


def softmax(a, axis=-1):
    a = _wrap(a)
    s = A.softmax(a.data, a.shape, axis=axis)
    out = Tensor(s, requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        g = out.grad
        if a.shape == ():
            a._accumulate(g)
            return
        if len(a.shape) == 1:
            dot = builtins.sum(g[i] * s[i] for i in range(len(s)))
            local = [s[i] * (g[i] - dot) for i in range(len(s))]
        else:
            m, n = a.shape
            local = []
            for r in range(m):
                dot = builtins.sum(g[r][c] * s[r][c] for c in range(n))
                local.append([s[r][c] * (g[r][c] - dot) for c in range(n)])
        a._accumulate(local)

    out._backward = _backward
    return out


# --------------------------------------------------------------------------- #
# Math                                                                         #
# --------------------------------------------------------------------------- #
def exp(a):
    a = _wrap(a)
    out = Tensor(A.exp(a.data, a.shape), requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        ga_full = A.broadcast_ew(operator.mul, out.grad, out.shape, out.data, a.shape)
        a._accumulate(_reduce_grad(ga_full, out.shape, a.shape))

    out._backward = _backward
    return out


def log(a):
    a = _wrap(a)
    out = Tensor(A.log(a.data, a.shape), requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        local = _recip(a.data, a.shape)
        ga_full = A.broadcast_ew(operator.mul, out.grad, out.shape, local, a.shape)
        a._accumulate(_reduce_grad(ga_full, out.shape, a.shape))

    out._backward = _backward
    return out


def sqrt(a):
    a = _wrap(a)
    out = Tensor(A.sqrt(a.data, a.shape), requires_grad=a.requires_grad)
    out._prev = {a}

    def _backward():
        local = A.ew(lambda v: 0.5 / math.sqrt(v) if v > 0 else 0.0, a.data, a.data) \
            if a.shape == () else _half_recip_sqrt(a.data, a.shape)
        ga_full = A.broadcast_ew(operator.mul, out.grad, out.shape, local, a.shape)
        a._accumulate(_reduce_grad(ga_full, out.shape, a.shape))

    out._backward = _backward
    return out


def _half_recip_sqrt(data, shape):
    if shape == ():
        return 0.5 / math.sqrt(data) if data > 0 else 0.0
    if len(shape) == 1:
        return [0.5 / math.sqrt(v) if v > 0 else 0.0 for v in data]
    return [[0.5 / math.sqrt(v) if v > 0 else 0.0 for v in row] for row in data]
