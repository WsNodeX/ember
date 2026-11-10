"""End-to-end integration tests: train real networks and check outcomes."""

import random
import unittest

import ember as eb
from ember import Tensor, nn, SGD, TensorDataset, DataLoader


class TestIntegration(unittest.TestCase):
    def test_xor_is_learnable(self):
        model = nn.Sequential(
            nn.Linear(2, 8, rng=random.Random(2)),
            nn.Tanh(),
            nn.Linear(8, 2, rng=random.Random(3)),
        )
        opt = SGD(model.parameters(), lr=0.1)
        xs = [Tensor([0.0, 0.0]), Tensor([0.0, 1.0]),
              Tensor([1.0, 0.0]), Tensor([1.0, 1.0])]
        ys = [0, 1, 1, 0]
        xb = Tensor([list(x.data) for x in xs])
        for _ in range(1500):
            pred = model(xb)
            loss = nn.CrossEntropyLoss()(pred, ys)
            loss.backward()
            opt.step()
            model.zero_grad()
        correct = 0
        for x, label in zip(xs, ys):
            out = model(x)
            pred_label = 0 if out.data[0] > out.data[1] else 1
            correct += pred_label == label
        self.assertEqual(correct, 4)

    def test_sequential_classifier_improves(self):
        # A trivially solvable 2-class problem.
        rng = random.Random(11)
        xs, ys = [], []
        for _ in range(120):
            c = rng.randint(0, 1)
            xs.append(Tensor([float(c) + rng.gauss(0, 0.05),
                              float(c) - rng.gauss(0, 0.05)]))
            ys.append(c)
        model = nn.Sequential(nn.Linear(2, 4, rng=random.Random(5)),
                              nn.Tanh(),
                              nn.Linear(4, 2, rng=random.Random(6)))
        opt = SGD(model.parameters(), lr=0.1)
        loader = DataLoader(TensorDataset(xs, ys), batch_size=20, shuffle=True,
                            rng=random.Random(7))
        for _ in range(120):
            for batch_x, batch_y in loader:
                pred = model(batch_x)
                loss = nn.CrossEntropyLoss()(pred, batch_y)
                loss.backward()
                opt.step()
                model.zero_grad()
        # evaluate on the same points (small problem)
        xb = Tensor([list(x.data) for x in xs])
        preds = model(xb)
        correct = sum(
            1 for i in range(len(ys))
            if (0 if preds.data[i][0] > preds.data[i][1] else 1) == ys[i]
        )
        self.assertGreaterEqual(correct, int(0.9 * len(ys)))


if __name__ == "__main__":
    unittest.main()
