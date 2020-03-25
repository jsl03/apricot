"""
Tests for the gp_internal (C++) kernels
"""

import unittest
import numpy as np
import gp_utils
from apricot.core import gp_internal


class TestKernels(unittest.TestCase):

    def setUp(self):
        amp = 1.0
        ls = np.array([0.25])
        sigma = 0.0
        jitter = 1e-10
        self.theta = (amp, ls, sigma, jitter)

    def test_eq(self):
        x_arr = np.linspace(0, 1, 10).reshape(-1, 1, order='F')
        amp, ls, sigma, jitter = self.theta
        k_r = gp_internal.cov_eq(
            x_arr,
            amp**2,
            ls**2,
            sigma**2 + jitter,
        )
        k_e = gp_utils.kernels.eq.covariance(
            x_arr,
            None,
            amp,
            ls,
            sigma,
            jitter
        )
        self.assertTrue(np.allclose(k_r, k_e))

    def test_m52(self):
        x_arr = np.linspace(0, 1, 10).reshape(-1, 1, order='F')
        amp, ls, sigma, jitter = self.theta
        k_r = gp_internal.cov_m52(
            x_arr,
            amp**2,
            ls**2,
            sigma**2 + jitter,
        )
        k_e = gp_utils.kernels.m52.covariance(
            x_arr,
            None,
            amp,
            ls,
            sigma,
            jitter
        )
        self.assertTrue(np.allclose(k_r, k_e))

    def test_m32(self):
        x_arr = np.linspace(0, 1, 10).reshape(-1, 1, order='F')
        amp, ls, sigma, jitter = self.theta
        k_r = gp_internal.cov_m32(
            x_arr,
            amp**2,
            ls**2,
            sigma**2 + jitter,
        )
        k_e = gp_utils.kernels.m32.covariance(
            x_arr,
            None,
            amp,
            ls,
            sigma,
            jitter
        )
        self.assertTrue(np.allclose(k_r, k_e))

if __name__ == '__main__':
    unittest.main()
