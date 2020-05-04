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


class LinearModel:
    """ Bayesian Linear Regression (QR parametrisation) """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        n_pts, index_dimension = x_data.shape
        data = {
            'x': x_data,
            'n': n_pts,
            'd': index_dimension,
            'y': y_data,
            'sigma_scale': y_data.std() / 10,
        }
        model = get_lreg_model()
        samples = model.sampling(data)
        self.alpha = samples['alpha']
        self.beta = samples['beta']
        self.sigma = samples['sigma']
        self.n_samples = self.alpha.shape[0]

    def expectation(self, x_star: np.ndarray):
        """ Posterior expectation"""
        y_pred = np.zeros(x_star.shape[0], order='C')
        for i in range(self.n_samples):
            y_pred += np.dot(x_star, self.beta[i, :]) + self.alpha[i]
        return y_pred / self.n_samples

    def predict(
            self,
            x_star: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """ Predictive distributions"""
        pred_means = np.empty((self.n_samples, x_star.shape[0]), order='C')
        for i in range(self.n_samples):
            pred_means[i, :] = np.dot(x_star, self.beta[i, :]) + self.alpha[i]
        return pred_means, self.sigma

    def lppd(self, x_star: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        """Log Pointwise Predictive Densities"""
        n_pts = y_star.shape[0]
        lppd = np.empty((self.n_samples, n_pts), order='C')
        for i in range(self.n_samples):
            pred_mean = np.dot(x_star, self.beta[i, :]) + self.alpha[i]
            pred_sd = self.sigma[i]
            lppd[i, :] = stats.norm.logpdf(
                y_star,
                loc=pred_mean,
                scale=np.full(n_pts, pred_sd)
            )
        return lppd
