"""Tests for the loss functions (analytic backward vs finite differences)."""

import math
import unittest

from ember import Tensor, nn
from tests.utils import grad_check


class TestLosses(unittest.TestCase):
    def test_mse_grad_check(self):
        def build(x):
            pred = Tensor([x[0], x[1], x[2]], requires_grad=True)
            target = Tensor([x[3], x[4], x[5]])
            return [pred, nn.MSELoss()(pred, target)]
        grad_check(build, [0.5, -0.3, 1.2, 0.0, 0.0, 0.0], msg="mse-pred")

    def test_mse_value(self):
        pred = Tensor([1.0, 2.0])
        target = Tensor([1.0, 0.0])
        # mean((0)^2 + (2)^2) = 2.0
        self.assertAlmostEqual(nn.MSELoss()(pred, target).item(), 2.0, places=6)

    def test_cross_entropy_grad_check(self):
        def build(x):
            a = Tensor([[x[0], x[1]], [x[2], x[3]]], requires_grad=True)
            return [a, nn.CrossEntropyLoss()(a, [0, 1])]
        grad_check(build, [0.2, -0.3, 0.5, 0.1], msg="cross-entropy")

    def test_cross_entropy_value(self):
        logits = Tensor([[math.log(0.8), math.log(0.2)]])
        # -log(0.8)
        self.assertAlmostEqual(
            nn.CrossEntropyLoss()(logits, [0]).item(), -math.log(0.8), places=6)

    def test_bce_grad_check(self):
        def build(x):
            a = Tensor([x[0], x[1]], requires_grad=True)
            t = Tensor([x[2], x[3]])
            return [a, nn.BCEWithLogitsLoss()(a, t)]
        grad_check(build, [0.2, -0.3, 1.0, 0.0], msg="bce")

    def test_bce_value(self):
        logits = Tensor([math.log(9.0)])  # sigmoid = 0.9
        target = Tensor([1.0])
        # -log(0.9)
        self.assertAlmostEqual(
            nn.BCEWithLogitsLoss()(logits, target).item(), -math.log(0.9), places=6)


if __name__ == "__main__":
    unittest.main()
