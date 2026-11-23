"""Benchmark: the same tiny MLP trained with Ember and with NumPy.

Same architecture (1-16-1 with ReLU), same data, same initial weights,
same plain-SGD updates, full batch. The only variable is the tensor
engine underneath: Ember's pure-Python autograd versus hand-written
backprop in NumPy.

Run::

    python benchmark.py

Requires NumPy only for the baseline column (Ember itself stays
zero-dependency).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random

import ember as eb
from ember import Tensor, nn, SGD

N_SAMPLES = 128
STEPS = 100
HIDDEN = 16
LR = 0.1


def build_data():
    rng = random.Random(0)
    xs, ys = [], []
    for _ in range(N_SAMPLES):
        x = rng.uniform(0.0, 1.0)
        xs.append([x])
        ys.append([0.8 * (2.0 * x - 1.0) ** 2 + 0.1])  # smooth 1-D target
    return xs, ys


def build_model():
    return nn.Sequential(
        nn.Linear(1, HIDDEN, rng=random.Random(1)),
        nn.ReLU(),
        nn.Linear(HIDDEN, 1, rng=random.Random(2)),
    )


def train_ember(xs, ys):
    model = build_model()
    opt = SGD(model.parameters(), lr=LR)
    x = Tensor(xs, requires_grad=False)
    y = Tensor(ys, requires_grad=False)
    loss_fn = nn.MSELoss()

    t0 = time.perf_counter()
    loss = None
    for _ in range(STEPS):
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        model.zero_grad()
    elapsed = time.perf_counter() - t0
    return elapsed, loss.item()


def train_numpy(xs, ys):
    import numpy as np

    model = build_model()
    # identical starting weights as the Ember run
    W1 = np.array(model.layers[0].weight.data)
    b1 = np.array(model.layers[0].bias.data)
    W2 = np.array(model.layers[2].weight.data)
    b2 = np.array(model.layers[2].bias.data)

    X = np.array(xs)
    Y = np.array(ys)

    t0 = time.perf_counter()
    loss = None
    for _ in range(STEPS):
        h = np.maximum(X @ W1 + b1, 0.0)
        pred = h @ W2 + b2
        loss = float(np.mean((pred - Y) ** 2))

        d_pred = 2.0 * (pred - Y) / (N_SAMPLES * 1)
        gW2 = h.T @ d_pred
        gb2 = d_pred.sum(axis=0)
        d_h = d_pred @ W2.T
        d_h[h <= 0.0] = 0.0
        gW1 = X.T @ d_h
        gb1 = d_h.sum(axis=0)

        W1 -= LR * gW1
        b1 -= LR * gb1
        W2 -= LR * gW2
        b2 -= LR * gb2
    elapsed = time.perf_counter() - t0
    return elapsed, loss


def main():
    import platform

    try:
        import numpy as _np
        numpy_version = _np.__version__
    except ImportError:
        print("numpy is not installed - the baseline column needs it.")
        print("    pip install numpy")
        return

    xs, ys = build_data()
    ember_time, ember_loss = train_ember(xs, ys)
    numpy_time, numpy_loss = train_numpy(xs, ys)

    print(f"task      : 1-{HIDDEN}-1 ReLU MLP, full batch, {STEPS} steps")
    print(f"python    : {platform.python_version()}")
    print(f"numpy     : {numpy_version} (baseline only)")
    print()
    print(f"{'engine':<10} {'time':>10} {'final loss':>12}")
    print(f"{'ember':<10} {ember_time:>9.2f}s {ember_loss:>12.6f}")
    print(f"{'numpy':<10} {numpy_time:>9.2f}s {numpy_loss:>12.6f}")
    print()
    print(f"speed-up  : numpy is {ember_time / numpy_time:.0f}x faster on this workload")
    print("both engines start from identical weights; final losses should")
    print("match to within floating-point noise.")


if __name__ == "__main__":
    main()
