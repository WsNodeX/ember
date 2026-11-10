"""Example: train a small MLP to solve the XOR problem with Ember.

Run::

    python examples/xor_mlp.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import ember as eb
from ember import Tensor, nn, SGD


def main():
    # XOR truth table
    xs = [Tensor([0.0, 0.0]), Tensor([0.0, 1.0]),
          Tensor([1.0, 0.0]), Tensor([1.0, 1.0])]
    ys = [0, 1, 1, 0]

    model = nn.Sequential(
        nn.Linear(2, 8, rng=random.Random(2)),
        nn.Tanh(),
        nn.Linear(8, 2, rng=random.Random(3)),
    )
    opt = SGD(model.parameters(), lr=0.1)
    xb = Tensor([list(x.data) for x in xs])  # batched (4, 2)

    for epoch in range(1500):
        pred = model(xb)
        loss = nn.CrossEntropyLoss()(pred, ys)
        loss.backward()
        opt.step()
        model.zero_grad()
        if epoch % 300 == 0 or epoch == 1499:
            print(f"epoch {epoch:4d}  loss={loss.item():.4f}")

    print("\nXOR predictions:")
    correct = 0
    for x, label in zip(xs, ys):
        out = model(x)
        pred_label = 0 if out.data[0] > out.data[1] else 1
        correct += pred_label == label
        print(f"  [{x.data[0]:.0f}, {x.data[1]:.0f}] -> "
              f"class {pred_label} (truth {label})")
    print(f"\naccuracy: {correct}/4")


if __name__ == "__main__":
    main()

