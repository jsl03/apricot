"""
DOCSTRING

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""

import typing
import functools
import pystan  # type: ignore
import numpy as np  # type: ignore
from scipy import stats  # type: ignore


LREG_QR = """
data {
    int<lower=0> n;
    int<lower=0> d;
    matrix[n, d] x;
    vector[n] y;
    real<lower=0> sigma_scale;
}
transformed data {
    matrix[n, d] Q_ast;
    matrix[d, d] R_ast;
    matrix[d, d] R_ast_inverse;
    Q_ast = qr_Q(x)[, 1:d] * sqrt(n - 1);
    R_ast = qr_R(x)[1:d, ] / sqrt(n - 1);
    R_ast_inverse = inverse(R_ast);
}
parameters {
    real alpha;
    vector[d] theta;
    real<lower=0> sigma;  // noise sd
}
model {
    sigma ~ normal(0, sigma_scale);
    y ~ normal(Q_ast * theta + alpha, sigma);
}
generated quantities {
    vector[d] beta;
    beta = R_ast_inverse * theta; // coefficients on x
}
"""


@functools.lru_cache(maxsize=1)
def get_lreg_model() -> pystan.StanModel:
    """ Return pyStan linear regression model """
    return pystan.StanModel(model_code=LREG_QR)


class LinearModel(object):
    """ Bayesian Linear Regression (QR parametrisation) """
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        n, d = x.shape
        data = {
            'x': x,
            'n': n,
            'd': d,
            'y': y,
            'sigma_scale': y.std() / 10,
        }
        model = get_lreg_model()
        samples = model.sampling(data)
        self.alpha = samples['alpha']
        self.beta = samples['beta']
        self.sigma = samples['sigma']
        self.m = self.alpha.shape[0]

    def expectation(self, xstar: np.ndarray):
        """ Posterior expectation"""
        ystar = np.zeros(xstar.shape[0], order='C')
        for i in range(self.m):
            ystar += np.dot(xstar, self.beta[i, :]) + self.alpha[i]
        return ystar / self.m

    def predict(self, xstar: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """ Predictive distributions"""
        mu = np.empty((self.m, xstar.shape[0]), order='C')
        for i in range(self.m):
            mu[i, :] = np.dot(xstar, self.beta[i, :]) + self.alpha[i]
        return mu, self.sigma

    def lppd(self, xstar: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log Pointwise Predictive Densities"""
        n = y.shape[0]
        lppd = np.empty((self.m, n), order='C')
        for i in range(self.m):
            mu = np.dot(xstar, self.beta[i, :]) + self.alpha[i]
            sigma = self.sigma[i]
            lppd[i, :] = stats.norm.logpdf(y, loc=mu, scale=np.full(n, sigma))
        return lppd
