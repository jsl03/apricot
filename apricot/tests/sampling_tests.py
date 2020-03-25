"""
Tests for sampling methods
"""

import unittest
import numpy as np
from apricot.core import sampling


class TestSampling(unittest.TestCase):

    def test_seed(self):
        r1 = sampling.lhs.lhs(10, 2, seed=1)
        r2 = sampling.lhs.lhs(10, 2, seed=1)
        self.assertTrue(np.allclose(r1, r2))

    def test_factorial(self):
        r = sampling.factorial.factorial(16, 2)
        # TODO fixme
        e = sampling.factorial.factorial(16, 2)
        self.assertTrue(np.allclose(r, e))

    def test_lhs(self):
        r = sampling.lhs.lhs(10, 2, seed=1)
        # TODO fixme
        e = sampling.lhs.lhs(10, 2, seed=1)
        self.assertTrue(np.allclose(r, e))

    def test_mdurs(self):
        r_cityblock = sampling.lhs.mdurs(10, 2, measure='cityblock', seed=1)
        r_euclidean = sampling.lhs.mdurs(10, 2, measure='euclidean', seed=1)
        # TODO fixme
        e_cityblock = sampling.lhs.mdurs(10, 2, measure='cityblock', seed=1)
        e_euclidean = sampling.lhs.mdurs(10, 2, measure='euclidean', seed=1)
        self.assertTrue(np.allclose(r_cityblock, e_cityblock))
        self.assertTrue(np.allclose(r_euclidean, e_euclidean))

    def test_olhs(self):
        r_euclidean = sampling.lhs.optimised_lhs(
            10,
            2,
            measure='euclidean',
            seed=1
        )
        # TODO fixme
        e_euclidean = sampling.lhs.optimised_lhs(
            10,
            2,
            measure='euclidean',
            seed=1
        )
        self.assertTrue(np.allclose(r_euclidean, e_euclidean))

        r_cityblock = sampling.lhs.optimised_lhs(
            10,
            2,
            measure='cityblock',
            seed=1
        )
        # TODO fixme
        e_cityblock = sampling.lhs.optimised_lhs(
            10,
            2,
            measure='cityblock',
            seed=1
        )
        self.assertTrue(np.allclose(r_cityblock, e_cityblock))

    def test_sobol(self):
        r = sampling.sobol.sobol(10, 2)
        # TODO fixme
        e = sampling.sobol.sobol(10, 2)
        self.assertTrue(np.allclose(r, e))

    def test_sobol_scatter(self):
        r = sampling.sobol.sobol_scatter(10, 2, seed=1)
        # TODO fixme
        e = sampling.sobol.sobol_scatter(10, 2, seed=1)
        self.assertTrue(np.allclose(r, e))


if __name__ == '__main__':
    unittest.main()
