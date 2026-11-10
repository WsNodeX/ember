"""Example: fit a tiny linear model ``y = 3x + 2`` with Ember's SGD.

Run::

    python examples/fit_linear.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import ember as eb
from ember import Tensor, nn, SGD, TensorDataset, DataLoader


def main():
    rng = random.Random(42)
    xs = [Tensor([rng.uniform(-2.0, 2.0)]) for _ in range(200)]
    ys = [Tensor([3.0 * x.data[0] + 2.0 + rng.gauss(0.0, 0.05)]) for x in xs]

    model = nn.Linear(1, 1, rng=random.Random(0))
    opt = SGD(model.parameters(), lr=0.05)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=True,
                        rng=random.Random(1))

    for epoch in range(200):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            loss = nn.MSELoss()(model(xb), yb)
            epoch_loss += loss.item()
            n_batches += 1
            loss.backward()
            opt.step()
            model.zero_grad()
        if epoch % 40 == 0 or epoch == 199:
            print(f"epoch {epoch:3d}  loss={epoch_loss / max(1, n_batches):.4f}")

    w = model.weight.data[0][0]
    b = model.bias.data[0]
    print(f"\nrecovered: y = {w:.3f} * x + {b:.3f}   (true: y = 3.000 * x + 2.000)")
    print(f"fit error on weight: {abs(w - 3.0):.4f}, on bias: {abs(b - 2.0):.4f}")


if __name__ == "__main__":
    main()
