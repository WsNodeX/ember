"""Gradient-check tests for every differentiable operation."""

import unittest

from ember import Tensor, ops
from tests.utils import grad_check


class TestOpGradients(unittest.TestCase):
    def test_add(self):
        def build(x):
            a = Tensor([x[0], x[1]], requires_grad=True)
            b = Tensor([x[2], x[3]], requires_grad=True)
            return [a, b, ops.sum(ops.mul(ops.add(a, b), ops.add(a, b)))]
        grad_check(build, [1.0, 2.0, 3.0, 4.0], msg="add")

    def test_sub(self):
        def build(x):
            a = Tensor([x[0], x[1]], requires_grad=True)
            b = Tensor([x[2], x[3]], requires_grad=True)
            return [a, b, ops.sum(ops.mul(ops.sub(a, b), ops.sub(a, b)))]
        grad_check(build, [1.0, 2.0, 3.0, 4.0], msg="sub")

    def test_mul(self):
        def build(x):
            a = Tensor([x[0], x[1]], requires_grad=True)
            b = Tensor([x[2], x[3]], requires_grad=True)
            return [a, b, ops.sum(ops.mul(ops.mul(a, b), ops.mul(a, b)))]
        grad_check(build, [1.0, 2.0, 3.0, 4.0], msg="mul")

    def test_div(self):
        def build(x):
            a = Tensor([x[0], x[1]], requires_grad=True)
            b = Tensor([x[2], x[3]], requires_grad=True)
            return [a, b, ops.sum(ops.mul(ops.div(a, b), ops.div(a, b)))]
        grad_check(build, [2.0, 4.0, 1.0, 3.0], msg="div")

    def test_scalar_matmul(self):
        def build(x):
            a = Tensor([[x[0], x[1]], [x[2], x[3]]], requires_grad=True)
            b = Tensor(2.0, requires_grad=True)
            return [a, b, ops.sum(ops.mul(ops.mul(a, b), ops.mul(a, b)))]
        grad_check(build, [0.5, 0.1, 0.2, 0.3], msg="scalar-matmul")

    def test_matmul_2d(self):
        def build(x):
            a = Tensor([[x[0], x[1]], [x[2], x[3]]], requires_grad=True)
            b = Tensor([[x[4], x[5]], [x[6], x[7]]], requires_grad=True)
            return [a, b, ops.sum(ops.mul(ops.matmul(a, b), ops.matmul(a, b)))]
        grad_check(build, [0.5, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.9], msg="matmul2d")

    def test_matmul_matvec(self):
        def build(x):
            a = Tensor([[x[0], x[1]], [x[2], x[3]]], requires_grad=True)
            b = Tensor([x[4], x[5]], requires_grad=True)
            return [a, b, ops.sum(ops.mul(ops.matmul(a, b), ops.matmul(a, b)))]
        grad_check(build, [0.5, 0.1, 0.2, 0.3, 0.4, 0.6], msg="matvec")

    def test_matmul_vecmat(self):
        def build(x):
            a = Tensor([x[0], x[1], x[2]], requires_grad=True)
            b = Tensor([[x[3], x[4]], [x[5], x[6]], [x[7], x[8]]], requires_grad=True)
            return [a, b, ops.sum(ops.mul(ops.matmul(a, b), ops.matmul(a, b)))]
        grad_check(build, [0.5, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], msg="vecmat")

    def test_matmul_dot(self):
        def build(x):
            a = Tensor([x[0], x[1], x[2]], requires_grad=True)
            b = Tensor([x[3], x[4], x[5]], requires_grad=True)
            c = ops.matmul(a, b)
            return [a, b, ops.mul(c, c)]
        grad_check(build, [0.5, 0.1, 0.2, 0.3, 0.4, 0.6], msg="dot")

    def test_bias_add(self):
        def build(x):
            a = Tensor([[x[0], x[1]], [x[2], x[3]]], requires_grad=True)
            b = Tensor([x[4], x[5]], requires_grad=True)
            return [a, b, ops.sum(ops.mul(ops.add(a, b), ops.add(a, b)))]
        grad_check(build, [1.0, 2.0, 3.0, 4.0, 0.5, 0.5], msg="bias-add")

    def test_relu(self):
        def build(x):
            a = Tensor([x[0], x[1], x[2]], requires_grad=True)
            return [a, ops.sum(ops.mul(a.relu(), a.relu()))]
        grad_check(build, [-1.0, 0.5, 2.0], msg="relu")

    def test_sigmoid(self):
        def build(x):
            a = Tensor([x[0], x[1]], requires_grad=True)
            return [a, ops.sum(ops.mul(a.sigmoid(), a.sigmoid()))]
        grad_check(build, [-2.0, 1.0], msg="sigmoid")

    def test_tanh(self):
        def build(x):
            a = Tensor([x[0], x[1], x[2]], requires_grad=True)
            return [a, ops.sum(ops.mul(a.tanh(), a.tanh()))]
        grad_check(build, [-1.0, 0.5, 2.0], msg="tanh")

    def test_softmax(self):
        def build(x):
            a = Tensor([x[0], x[1], x[2]], requires_grad=True)
            return [a, ops.sum(ops.mul(a.softmax(), a.softmax()))]
        grad_check(build, [1.0, 2.0, 3.0], msg="softmax")

    def test_exp(self):
        def build(x):
            a = Tensor([x[0], x[1]], requires_grad=True)
            return [a, ops.sum(ops.mul(a.exp(), a.exp()))]
        grad_check(build, [0.3, 0.7], msg="exp")

    def test_pow(self):
        def build(x):
            a = Tensor([x[0], x[1]], requires_grad=True)
            return [a, ops.sum(ops.mul(ops.pow(a, 3.0), ops.pow(a, 3.0)))]
        grad_check(build, [1.0, 2.0], msg="pow3")

    def test_transpose_grad(self):
        def build(x):
            a = Tensor([[x[0], x[1]], [x[2], x[3]]], requires_grad=True)
            return [a, ops.sum(ops.mul(a.transpose(), a.transpose()))]
        grad_check(build, [0.5, 0.1, 0.2, 0.3], msg="transpose")

    def test_sum_grad(self):
        def build(x):
            a = Tensor([[x[0], x[1]], [x[2], x[3]]], requires_grad=True)
            return [a, ops.mul(a.sum(), a.sum())]
        grad_check(build, [0.5, 0.1, 0.2, 0.3], msg="sum")

    def test_mean_grad(self):
        def build(x):
            a = Tensor([[x[0], x[1]], [x[2], x[3]]], requires_grad=True)
            return [a, ops.mul(a.mean(), a.mean())]
        grad_check(build, [0.5, 0.1, 0.2, 0.3], msg="mean")


if __name__ == "__main__":
    unittest.main()
