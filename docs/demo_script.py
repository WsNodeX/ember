"""Train a 2-8-2 MLP on XOR with Ember.

This file is executed twice from the same source: by
``python docs/demo_script.py`` on your machine, and inside the browser demo
(Pyodide loads this exact text). The code block on the demo page is this file.
"""

import json
import random

from ember import Tensor, nn, SGD

XOR_X = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_Y = [0, 1, 1, 0]

xb = Tensor([list(p) for p in XOR_X])
model = nn.Sequential(
    nn.Linear(2, 8, rng=random.Random(2)),
    nn.Tanh(),
    nn.Linear(8, 2, rng=random.Random(3)),
)
opt = SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
state = {"epoch": 0, "loss": 0.0, "curve": []}


def train(steps=1):
    """Run `steps` optimiser updates, return progress as JSON."""
    for _ in range(steps):
        loss = loss_fn(model(xb), XOR_Y)
        loss.backward()
        opt.step()
        model.zero_grad()
        state["epoch"] += 1
        state["loss"] = loss.item()
    state["curve"].append(round(state["loss"], 4))
    return json.dumps({"epoch": state["epoch"], "loss": round(state["loss"], 4)})


def classify():
    """Classify all four inputs, return the table as JSON."""
    rows, correct = [], 0
    for (a, b), truth in zip(XOR_X, XOR_Y):
        probs = model(Tensor([a, b])).softmax().data
        guess = 0 if probs[0] > probs[1] else 1
        correct += 1 if guess == truth else 0
        rows.append({"x": [a, b], "p0": round(probs[0], 3),
                     "p1": round(probs[1], 3), "guess": guess, "truth": truth})
    return json.dumps({"rows": rows, "correct": correct, "total": len(XOR_Y)})


if __name__ == "__main__":
    while state["epoch"] < 1500:
        info = json.loads(train(50))
        if info["epoch"] % 300 == 0:
            print(f"epoch {info['epoch']:4d}  loss={info['loss']:.4f}")

    result = json.loads(classify())
    for row in result["rows"]:
        print(f"  [{row['x'][0]:.0f}, {row['x'][1]:.0f}] -> "
              f"class {row['guess']} (truth {row['truth']})")
    print(f"accuracy: {result['correct']}/{result['total']}")
