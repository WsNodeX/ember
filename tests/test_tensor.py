"""Tests for the :class:`ember.Tensor` autograd engine."""

import unittest

from ember import Tensor, ops


class TestTensor(unittest.TestCase):
    def test_scalar_item(self):
        t = Tensor(2.5)
        self.assertEqual(t.shape, ())
        self.assertEqual(t.item(), 2.5)

    def test_shape_and_requires_grad(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        self.assertEqual(t.shape, (2, 2))
        self.assertTrue(t.requires_grad)

    def test_topo_order(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = ops.add(a, b)
        d = ops.mul(c, c)
        d.backward()
        # d/dc = 2c ; d/da = d/db = 2c
        self.assertEqual(a.grad, [8.0, 12.0])
        self.assertEqual(b.grad, [8.0, 12.0])

    def test_zero_grad(self):
        a = Tensor([1.0], requires_grad=True)
        b = ops.mul(a, a)
        b.backward()
        self.assertIsNotNone(a.grad)
        a.zero_grad()
        self.assertIsNone(a.grad)

    def test_intermediate_grad_not_kept_by_default(self):
        a = Tensor([1.0], requires_grad=True)
        const = Tensor([1.0])  # leaf without requires_grad
        b = ops.add(a, const)
        c = ops.mul(b, b)
        c.backward()
        # The constant leaf is seeded to zeros by backward but never receives a
        # real gradient (it does not require one); the leaf that requires grad
        # gets the correct value.
        self.assertEqual(const.grad, [0.0])
        # d/da (a+1)^2 at a=1 is 2*(1+1) = 4
        self.assertEqual(a.grad, [4.0])

    def test_chain_rule_two_paths(self):
        # x used in two branches: y = (x+1)*(x+2)
        x = Tensor(3.0, requires_grad=True)
        y = ops.mul(ops.add(x, 1.0), ops.add(x, 2.0))
        y.backward()
        # dy/dx = (x+2) + (x+1) = 5 + 4 = 9
        self.assertAlmostEqual(x.grad, 9.0, places=6)

    def test_neg_and_sub(self):
        a = Tensor([2.0, 5.0], requires_grad=True)
        b = ops.sub(a, Tensor([1.0, 1.0]))
        c = ops.neg(b)
        c.backward()
        self.assertEqual(a.grad, [-1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
