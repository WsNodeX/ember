"""Tests for the neural-network building blocks."""

import unittest

import ember as eb
from ember import Tensor, ops, nn
from tests.utils import grad_check, flatten
import math


class TestNN(unittest.TestCase):
    def test_linear_forward_shape(self):
        lin = nn.Linear(3, 4, rng=__import__("random").Random(0))
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = lin(x)
        self.assertEqual(y.shape, (2, 4))
        self.assertEqual(len(lin.parameters()), 2)

    def test_linear_grad_check(self):
        def build(x):
            lin = nn.Linear(2, 3, rng=__import__("random").Random(99))
            # overwrite weights/biases with the flat vector
            lin.weight.data = [[x[0], x[1], x[2]], [x[3], x[4], x[5]]]
            lin.bias.data = [x[6], x[7], x[8]]
            inp = Tensor([[0.5, 0.2], [0.1, 0.3]])
            y = lin(inp)
            return [lin.weight, lin.bias, ops.sum(ops.mul(y, y))]
        grad_check(build, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0], msg="linear")

    def test_activations_forward(self):
        x = Tensor([-1.0, 0.0, 2.0])
        self.assertEqual(nn.ReLU()(x).data, [0.0, 0.0, 2.0])
        self.assertAlmostEqual(nn.Sigmoid()(x).data[2], 1.0 / (1 + math.exp(-2.0)), places=6)
        self.assertAlmostEqual(nn.Tanh()(Tensor(0.0)).item(), 0.0, places=6)

    def test_module_parameters_recurses(self):
        model = nn.Sequential(nn.Linear(2, 3, rng=__import__("random").Random(0)),
                              nn.Tanh(),
                              nn.Linear(3, 1, rng=__import__("random").Random(1)))
        # 2 layers -> 4 parameter tensors
        self.assertEqual(len(model.parameters()), 4)

    def test_zero_grad_clears(self):
        lin = nn.Linear(2, 2, rng=__import__("random").Random(0))
        x = Tensor([[1.0, 1.0]])
        nn.MSELoss()(lin(x), Tensor([[0.0, 0.0]])).backward()
        self.assertTrue(any(p.grad is not None for p in lin.parameters()))
        lin.zero_grad()
        self.assertTrue(all(p.grad is None for p in lin.parameters()))

    def test_sequential_apply(self):
        seq = nn.Sequential(nn.Linear(2, 2, rng=__import__("random").Random(0)),
                            nn.ReLU())
        x = Tensor([[-1.0, 1.0]])
        out = seq(x)
        self.assertEqual(out.shape, (1, 2))
        self.assertEqual(out.data[0][0], 0.0)  # ReLU zeroes negative


if __name__ == "__main__":
    unittest.main()
