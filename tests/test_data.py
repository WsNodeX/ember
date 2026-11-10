"""Tests for the dataset / data-loader utilities."""

import random
import unittest

from ember import Tensor, TensorDataset, DataLoader


class TestData(unittest.TestCase):
    def test_dataset_length_and_getitem(self):
        xs = [Tensor(float(i)) for i in range(5)]
        ys = [Tensor(float(i * 2)) for i in range(5)]
        ds = TensorDataset(xs, ys)
        self.assertEqual(len(ds), 5)
        x, y = ds[2]
        self.assertEqual(x.item(), 2.0)
        self.assertEqual(y.item(), 4.0)

    def test_dataloader_batch_stacking(self):
        xs = [Tensor([float(i), float(i + 1)]) for i in range(10)]
        ys = [Tensor(float(i)) for i in range(10)]
        loader = DataLoader(TensorDataset(xs, ys), batch_size=4, shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 3)  # 4,4,2
        xb, yb = batches[0]
        self.assertEqual(xb.shape, (4, 2))
        self.assertEqual(yb.shape, (4,))

    def test_dataloader_shuffle_full_coverage(self):
        xs = [Tensor(float(i)) for i in range(8)]
        loader = DataLoader(TensorDataset(xs), batch_size=2, shuffle=True,
                           rng=random.Random(3))
        seen = []
        for (xb,) in loader:
            seen.extend(xb.data)
        self.assertEqual(sorted(seen), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_dataloader_empty_raises(self):
        with self.assertRaises(ValueError):
            TensorDataset()


if __name__ == "__main__":
    unittest.main()
