"""
Tests for the Emulator class.

TODO:
    Derivative tests
    Analytical loo-cv
    Analytical first order sobol indices
"""

import unittest
import numpy as np
import gp_utils
from apricot.core import exceptions
from apricot.core import emulator


def f_1d(x):
    """Arbitrary test function."""
    return x * np.sin(x * np.pi * 4)


class TestEmulator(unittest.TestCase):

    def setUp(self):
        x_1d = np.linspace(0, 1, 10).reshape(-1, 1)
        y_1d = f_1d(x_1d.ravel())
        hyperparameters_1d = {
            'amp': np.array([1.0]),
            'ls': np.array([[0.25]]),
            'xi': np.array([0.0])
        }
        self.model = emulator.Emulator(
            x_1d,
            y_1d,
            hyperparameters_1d,
            {},
        )
        self.x_test = x_1d
        self.y_test = y_1d
        self.theta_test = (
            hyperparameters_1d['amp'][0],
            hyperparameters_1d['ls'][0],
            hyperparameters_1d['xi'][0],
        )

    def test_format_input(self):
        r_1d = emulator._format_input(np.zeros(10), 1)
        e_1d = np.zeros((10, 1))
        r_2d = emulator._format_input(np.zeros((10, 2)), 2)
        e_2d = np.zeros((10, 2))
        self.assertTrue(np.allclose(r_1d, e_1d))
        self.assertTrue(np.allclose(r_2d, e_2d))
        with self.assertRaises(exceptions.ShapeError):
            emulator._format_input(np.zeros(10), 2)

    def test_marginals(self):
        xstar = np.linspace(0, 1, 100).reshape(-1, 1)
        r_m, r_v = self.model.marginals(xstar)
        # TODO: requires gp_utils; should be included in test folder
        e_m, e_v = gp_utils.gp.conditional_marginals(
            xstar,
            self.x_test,
            self.y_test,
            gp_utils.mean_functions.zero_mean,
            (),
            gp_utils.kernels.eq.covariance,
            self.theta_test
        )
        self.assertTrue(np.allclose(r_m.ravel(), e_m))
        self.assertTrue(np.allclose(r_v.ravel(), e_v))

    def test_posterior(self):
        xstar = np.linspace(0, 1, 100).reshape(-1, 1)
        r_m, r_k = self.model.posterior(xstar)
        # TODO: requires gp_utils; should be included in test folder
        e_m, e_k = gp_utils.gp.conditional_distribution(
            xstar,
            self.x_test,
            self.y_test,
            gp_utils.mean_functions.zero_mean,
            (),
            gp_utils.kernels.eq.covariance,
            self.theta_test
        )
        self.assertTrue(np.allclose(r_m.ravel(), e_m))
        self.assertTrue(np.allclose(r_k[0], e_k))

    def test_expectation(self):
        xstar = np.linspace(0, 1, 100).reshape(-1, 1)
        xstar_1d = np.array([0.1])
        r = self.model.expectation(xstar)
        r_1d, r_jac = self.model._gp.E_jac(xstar_1d)
        # TODO: requires gp_utils; should be included in test folder
        e = gp_utils.gp.predictive_mean(
            xstar,
            self.x_test,
            self.y_test,
            gp_utils.mean_functions.zero_mean,
            (),
            gp_utils.kernels.eq.covariance,
            self.theta_test
        )

        # TODO: test derivatives are broken;
        # compare to finite difference?
        e_1d, e_jac = self.model._gp.E_jac(xstar_1d)
        self.assertTrue(np.allclose(r, e))
        self.assertTrue(np.allclose(r_1d, e_1d))
        self.assertTrue(np.allclose(r_jac, e_jac))

    def test_px(self):
        xstar = np.linspace(0, 1, 100).reshape(-1, 1)
        xstar_1d = np.array([0.1])
        r = self.model.px(xstar)
        r_1d, r_jac = self.model._gp.px_jac(xstar_1d)
        # TODO: requires gp_utils; should be included in test folder
        # note PX is the negative predicted variance
        e = gp_utils.gp.predictive_variance(
            xstar,
            self.x_test,
            gp_utils.kernels.eq.covariance,
            self.theta_test
        ) * -1
        # TODO: test derivatives are broken;
        # compare to finite difference?
        e_1d, e_jac = self.model._gp.px_jac(xstar_1d)
        self.assertTrue(np.allclose(r, e))
        # self.assertTrue(np.allclose(r_1d, e_1d))
        # self.assertTrue(np.allclose(r_jac, e_jac))

    def test_ei(self):
        xstar = np.linspace(0, 1, 100).reshape(-1, 1)
        xstar_1d = np.array([0.1])
        r = self.model.ei(xstar)
        r_1d, r_jac = self.model._gp.ei_jac(xstar_1d)
        # TODO: requires gp_utils; should be included in test folder
        # note ei produced by apricot is supposed to be negative
        e = gp_utils.acquisition.expected_improvement(
            xstar,
            self.x_test,
            self.y_test,
            gp_utils.mean_functions.zero_mean,
            (),
            gp_utils.kernels.eq.covariance,
            self.theta_test
        ) * -1
        # TODO: test derivatives are broken;
        # compare to finite difference?
        e_1d, e_jac = self.model._gp.ei_jac(xstar_1d)
        self.assertTrue(np.allclose(r, e))
        # self.assertTrue(np.allclose(r_1d, e_1d))
        # self.assertTrue(np.allclose(r_jac, e_jac))

    def test_ucb(self):
        xstar = np.linspace(0, 1, 100).reshape(-1, 1)
        xstar_1d = np.array([0.1])
        beta = 3.0
        r = self.model.ucb(xstar, beta)
        r_1d, r_jac = self.model._gp.ucb_jac(xstar_1d, beta)
        e = self.model.ucb(xstar, beta)
        # TODO: requires gp_utils; should be included in test folder
        e = gp_utils.acquisition.upper_confidence_bound(
            xstar,
            self.x_test,
            self.y_test,
            gp_utils.mean_functions.zero_mean,
            (),
            gp_utils.kernels.eq.covariance,
            self.theta_test,
            beta,
        )
        # TODO: test derivatives are broken;
        # compare to finite difference?
        e_1d, e_jac = self.model._gp.ucb_jac(xstar_1d, beta)
        self.assertTrue(np.allclose(r, e))
        # self.assertTrue(np.allclose(r_1d, e_1d))
        # self.assertTrue(np.allclose(r_jac, e_jac))

    def test_loocv(self):
        r = self.model.loo_cv()
        # TODO better to use analytical test case here.
        e = self.model.loo_cv()
        self.assertTrue(np.allclose(r, e))

    def test_sobol_indices(self):
        r = self.model.sobol1()
        # TODO better to use analytical test case here.
        e = self.model.sobol1()
        self.assertTrue(np.allclose(r, e))


if __name__ == '__main__':
    unittest.main()
