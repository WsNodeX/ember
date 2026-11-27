# ember — full guide

`README.md` is deliberately short. This page holds the detail: what the engine
does, what the output actually looks like, and where the sharp edges are.

## What it is

Reverse-mode autograd, layers, losses, an SGD optimizer and a data loader — built
on nothing but the Python standard library. Written to be read end to end.

- `ember/` — 13 files, 1,450 lines.
- Every op's forward math and its backward rule live in the same file, next to
  each other, in plain Python.

## Quick start

```bash
git clone https://github.com/WsNodeX/ember && cd ember
python examples/fit_linear.py
```

Real output from `python examples/fit_linear.py` (Python 3.13.0, Windows):

```text
epoch   0  loss=4.4556
epoch  40  loss=0.0029
epoch  80  loss=0.0028
epoch 120  loss=0.0029
epoch 160  loss=0.0027
epoch 199  loss=0.0030

recovered: y = 3.000 * x + 2.002   (true: y = 3.000 * x + 2.000)
fit error on weight: 0.0004, on bias: 0.0022
```

Five lines, inline:

```python
import ember as eb

x = eb.Tensor([2.0])
y = (x * x).sum()
y.backward()
print(x.grad)          # [4.0]
```

XOR needs a hidden layer and a non-linearity:

```bash
python examples/xor_mlp.py
```

```text
epoch    0  loss=0.8014
epoch 1499  loss=0.0085

XOR predictions:
  [0, 0] -> class 0 (truth 0)
  [0, 1] -> class 1 (truth 1)
  [1, 0] -> class 1 (truth 1)
  [1, 1] -> class 0 (truth 0)accuracy: 4/4
```

## Why this exists

`micrograd` teaches backprop on scalars. Ember is the next size up: the tensor
supports matrices, so `Linear(2, 8)` and `CrossEntropyLoss` actually work, and
there is a real optimizer and a real data loader.

The tradeoff is that it is slow and it will stay slow — every op is a Python loop.
That is the point. It is not a PyTorch competitor; it is the thing PyTorch is made
of, small enough to hold in your head. If you need speed, use PyTorch. If you want
to stop treating `.backward()` as magic, read this.

## API

| Name | What it is |
|---|---|
| `Tensor` | Differentiable scalar, vector or matrix — `.data`, `.grad`, `.backward()` |
| `nn.Linear / ReLU / Sigmoid / Tanh / Softmax` | Layers and activations |
| `nn.Sequential` | Compose modules |
| `nn.MSELoss / CrossEntropyLoss / BCEWithLogitsLoss` | Losses with analytic backward |
| `SGD` | Momentum, Nesterov, L2 weight decay |
| `TensorDataset / DataLoader` | Mini-batching with shuffling |
| `ember.array` | The pure-Python tensor backend, if you want to poke at it |

## Tests

```bash
python -m unittest discover -s tests
# Ran 60 tests in 1.283s
# OK
```

CI runs this exact suite on Python 3.10–3.13, so the CI badge is live, not
decorative. The tests are plain `unittest`, so `python -m pytest -q` collects the
same 60.

Every operation is gradient-checked against central finite differences
(`tests/utils.py`). If a backward pass is wrong, a test fails — that is the whole
verification story, and it is why the engine is trustworthy enough to learn from.

## Benchmark

`benchmark.py` trains the same 1-16-1 ReLU MLP with both engines — identical
weights, identical data, identical updates — and times them. Real output
(Python 3.13.0, NumPy 2.2.5):

```text
engine     time   final loss
ember      0.64s     0.057455
numpy      0.01s     0.057455
```

NumPy is ~110× faster here, and that gap is the honest price of pure-Python
loops. The part that matters is the loss column: both engines reach the same
value to six decimal places, because they are running the same math. NumPy is
only needed for the baseline column — Ember itself installs nothing.

## Browser demo

<https://wsnodex.github.io/ember/> runs the real library under Pyodide inside the
tab. The page and the local script share one source file (`docs/demo_script.py`),
and `tools/build_pages.py` assembles the site from the working tree at deploy
time, so the demo cannot drift from the package.

## Limitations

- **Pure Python loops.** No NumPy, so matmul is a triple nested loop. It is slow,
  and because the whole point is readability it stays slow.
- **No GPU, no vectorisation, no plans for either.** Adding them means adding
  NumPy, which defeats the premise.
- **Not thread-safe.** The engine assumes a single writer.
- **No convolutions or RNNs.** `Linear`, activations and losses only.

## License

MIT. Contributions: see `CONTRIBUTING.md`.
