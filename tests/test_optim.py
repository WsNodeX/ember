"""Tests for the SGD optimizer and a real optimization loop."""

import random
import unittest

import ember as eb
from ember import Tensor, nn, SGD, TensorDataset, DataLoader


def make_data(n=200, seed=42):
    rng = random.Random(seed)
    xs = [Tensor([rng.uniform(-2, 2)]) for _ in range(n)]
    ys = [Tensor([3.0 * x.data[0] + 2.0 + rng.gauss(0, 0.05)]) for x in xs]
    return xs, ys


class TestOptim(unittest.TestCase):
    def test_linear_fit_recovers_params(self):
        xs, ys = make_data()
        lin = nn.Linear(1, 1, rng=random.Random(0))
        opt = SGD(lin.parameters(), lr=0.05)
        loader = DataLoader(TensorDataset(xs, ys), batch_size=32, shuffle=True,
                            rng=random.Random(1))
        for _ in range(200):
            for xb, yb in loader:
                loss = nn.MSELoss()(lin(xb), yb)
                loss.backward()
                opt.step()
                lin.zero_grad()
        w = lin.weight.data[0][0]
        b = lin.bias.data[0]
        self.assertAlmostEqual(w, 3.0, places=1)
        self.assertAlmostEqual(b, 2.0, places=1)

    def test_loss_decreases(self):
        xs, ys = make_data(n=50)
        lin = nn.Linear(1, 1, rng=random.Random(0))
        opt = SGD(lin.parameters(), lr=0.05)
        losses = []
        loader = DataLoader(TensorDataset(xs, ys), batch_size=50,
                           rng=random.Random(2))
        for _ in range(25):
            for xb, yb in loader:
                loss = nn.MSELoss()(lin(xb), yb)
                losses.append(loss.item())
                loss.backward()
                opt.step()
                lin.zero_grad()
        self.assertLess(losses[-1], losses[0])

    def test_weight_decay_changes_update(self):
        p = Tensor([1.0], requires_grad=True)
        opt = SGD([p], lr=0.1, weight_decay=1.0)
        p.grad = [0.5]
        opt.step()
        # update = lr*(grad + wd*p) = 0.1*(0.5 + 1.0*1.0) = 0.15
        self.assertAlmostEqual(p.data[0], 1.0 - 0.15, places=6)

    def test_momentum_velocity(self):
        p = Tensor([0.0], requires_grad=True)
        opt = SGD([p], lr=0.1, momentum=0.9)
        p.grad = [1.0]
        opt.step()  # v = 0.9*0 - 0.1*1 = -0.1 ; x += -0.1
        first = p.data[0]
        p.grad = [1.0]
        opt.step()  # v = 0.9*(-0.1) - 0.1*1 = -0.19 ; x += -0.19
        self.assertAlmostEqual(first, -0.1, places=6)
        self.assertAlmostEqual(p.data[0], -0.29, places=6)


if __name__ == "__main__":
    unittest.main()
