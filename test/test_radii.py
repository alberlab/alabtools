"""Isolated arithmetic regression for compute_radii; needs only NumPy.

Compile the complete function from source so optional compiled dependencies do
not prevent this test from running. Run: python test/test_radii.py
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import numpy as np


class IndexStub:
    def __init__(self, start, end):
        self.start = np.array(start, dtype=np.int32)
        self.end = np.array(end, dtype=np.int32)

    def __iter__(self):
        return (SimpleNamespace(start=s, end=e) for s, e in zip(self.start, self.end))


class TestComputeRadii(unittest.TestCase):
    def test_lengths_and_occupancy_without_overflow(self):
        source = Path(__file__).resolve().parents[1] / "alabtools" / "utils.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_radii")
        namespace = dict(np=np, _ftpi=4.0/3.0*np.pi, RADII_DTYPE=np.float32,
                         DEFAULT_NUCLEAR_VOLUME=4.0/3.0*np.pi*5000.0**3)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        cases = {
            "small_irregular": ([0, 0, 50], [100, 250, 750]),
            "above_int32_total": ([0, 0], [1_500_000_000, 1_000_000_000]),
            "above_uint32_irregular": ([0, 0, 0], [1_500_000_000, 2_000_000_000, 1_000_000_000]),
            "endpoint_subtraction": ([-2_000_000_000, 0], [2_000_000_000, 100]),
            "reported_run": ([0] * 10397, [500_000] * 10397),
        }

        for name, (starts, ends) in cases.items():
            with self.subTest(name=name):
                index = IndexStub(starts, ends)
                before = (index.start.copy(), index.end.copy())
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    radii = namespace["compute_radii"](index, occupancy=0.2, volume=4.0/3.0*np.pi*7000.0**3)
                lengths = [int(e) - int(s) for s, e in zip(starts, ends)]
                total = sum(lengths)  # Python integer reference, independent of NumPy promotion.
                expected = np.array([(0.2 * 7000.0**3 * b / total)**(1.0/3.0) for b in lengths])
                np.testing.assert_allclose(radii, expected, rtol=1e-6)
                self.assertEqual(radii.dtype, np.dtype("float32"))
                self.assertAlmostEqual(float(np.sum(radii.astype(np.float64)**3) / 7000.0**3), 0.2, places=6)
                self.assertEqual(index.start.dtype, np.dtype("int32"))
                self.assertEqual(index.end.dtype, np.dtype("int32"))
                np.testing.assert_array_equal(index.start, before[0])
                np.testing.assert_array_equal(index.end, before[1])
                if name == "reported_run":
                    self.assertEqual(total, 5_198_500_000)
                    self.assertAlmostEqual(float(expected[0]), 187.5593377290677, places=9)


if __name__ == "__main__":
    unittest.main()
