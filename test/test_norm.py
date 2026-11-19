import unittest
import warnings

import numpy as np

from alabtools.norm import bnewt


class DenseMatrix:
    """Provide bnewt's matrix interface without requiring the MKL backend."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape

    def dotv(self, vector):
        return self.values @ vector


class TestBnewt(unittest.TestCase):
    def test_balancing_with_and_without_masked_bins(self):
        cases = [
            ([[4, 1, 2], [1, 5, 1], [2, 1, 3]], []),
            ([[4, 1, 0], [1, 5, 0], [0, 0, 0]], [2]),
        ]
        for values, mask in cases:
            with self.subTest(mask=mask):
                matrix = DenseMatrix(values)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", DeprecationWarning)
                    bias = bnewt(matrix, mask=mask)

                self.assertEqual(bias.shape, (len(values), 1))
                self.assertTrue(np.all(np.isfinite(bias)))
                expected = np.ones_like(bias)
                expected[mask] = 0
                np.testing.assert_allclose(bias * matrix.dotv(bias), expected,
                                           rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
