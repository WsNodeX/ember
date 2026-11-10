"""Tests for the pure-Python array math backend."""

import math
import unittest

from ember import _array as A


class TestArrayMath(unittest.TestCase):
    def test_shape_inference(self):
        self.assertEqual(A.shape_of(3.0), ())
        self.assertEqual(A.shape_of([1.0, 2.0, 3.0]), (3,))
        self.assertEqual(A.shape_of([[1.0, 2.0], [3.0, 4.0]]), (2, 2))
        self.assertEqual(A.shape_of([]), (0,))

    def test_matmul_shapes(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        c, sh = A.matmul(a, (2, 2), b, (2, 2))
        self.assertEqual(sh, (2, 2))
        self.assertEqual(c, [[19.0, 22.0], [43.0, 50.0]])

    def test_matmul_matvec(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [5.0, 6.0]
        c, sh = A.matmul(a, (2, 2), b, (2,))
        self.assertEqual(sh, (2,))
        self.assertEqual(c, [17.0, 39.0])

    def test_matmul_vecm(self):
        a = [1.0, 2.0]
        b = [[3.0, 4.0], [5.0, 6.0]]
        c, sh = A.matmul(a, (2,), b, (2, 2))
        self.assertEqual(sh, (2,))
        self.assertEqual(c, [13.0, 16.0])

    def test_matmul_dot(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        c, sh = A.matmul(a, (3,), b, (3,))
        self.assertEqual(sh, ())
        self.assertEqual(c, 32.0)

    def test_transpose(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        t, sh = A.transpose(a, (2, 3))
        self.assertEqual(sh, (3, 2))
        self.assertEqual(t, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_sum_axis(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        self.assertEqual(A.sum_axis(a, (2, 2), axis=0), [4.0, 6.0])
        self.assertEqual(A.sum_axis(a, (2, 2), axis=1), [3.0, 7.0])
        self.assertEqual(A.sum_all(a), 10.0)

    def test_softmax_stable(self):
        s = A.softmax([1000.0, 1001.0, 1002.0], (3,))
        self.assertAlmostEqual(sum(s), 1.0, places=6)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in s))

    def test_relu(self):
        self.assertEqual(A.relu([-1.0, 0.0, 2.0], (3,)), [0.0, 0.0, 2.0])

    def test_sigmoid_bounds(self):
        s = A.sigmoid([-10.0, 0.0, 10.0], (3,))
        self.assertAlmostEqual(s[0], 0.0, places=4)
        self.assertAlmostEqual(s[1], 0.5, places=6)
        self.assertAlmostEqual(s[2], 1.0, places=4)

    def test_exp_log_roundtrip(self):
        x = [0.5, 1.0, 2.0]
        self.assertAlmostEqual(A.log(A.exp(x, (3,)), (3,))[1], 1.0, places=6)

    def test_broadcast_add(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [10.0, 20.0]
        out = A.broadcast_ew(lambda p, q: p + q, a, (2, 2), b, (2,))
        self.assertEqual(out, [[11.0, 22.0], [13.0, 24.0]])


if __name__ == "__main__":
    unittest.main()
