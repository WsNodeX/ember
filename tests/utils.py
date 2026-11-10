"""Shared helpers for the *ember* test-suite (not a test module itself)."""

import math

from ember import _array as A
from ember.tensor import Tensor


def flatten(d):
    out = []

    def rec(x):
        if A.is_scalar(x):
            out.append(x)
        else:
            for v in x:
                rec(v)

    rec(d)
    return out


def unflatten(flat, shape):
    if shape == ():
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    out = []
    idx = 0
    for r in range(shape[0]):
        out.append(list(flat[idx:idx + shape[1]]))
        idx += shape[1]
    return out


def num_grad(build, loss_fn, x0, eps=1e-5):
    """Central finite differences of the scalar loss over a flat vector."""
    out = []
    for i in range(len(x0)):
        xp = list(x0)
        xm = list(x0)
        xp[i] += eps
        xm[i] -= eps
        out.append((loss_fn(xp) - loss_fn(xm)) / (2 * eps))
    return out


def grad_check(build, x0, eps=1e-5, tol=1e-3, msg=""):
    """Compare analytic autograd gradients to finite differences.

    ``build`` maps a flat list of floats to ``[tensor0, tensor1, ..., loss]``.
    """
    res = build(list(x0))
    tensors, loss = res[:-1], res[-1]
    loss.backward()
    analytic = []
    for t in tensors:
        analytic.extend(flatten(t.grad))
    loss_fn = lambda v: build(v)[-1].item()
    numeric = num_grad(build, loss_fn, x0, eps)
    for a, n in zip(analytic, numeric):
        if abs(a - n) > tol:
            raise AssertionError(
                f"{msg or 'grad_check'} exceeded tol: analytic={a} numeric={n} "
                f"diff={abs(a - n)}"
            )
    return True
