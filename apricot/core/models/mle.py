# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
import numpy as np
from apricot.core.gp_internal import make_mle_objective
from apricot.core.logger import get_logger
from scipy import optimize


logger = get_logger()


def run_mle(interface_instance, x, y, jitter):
    # TODO kernel needs to be included / warning if kernel != 'eq'
    noise_type, xi = interface_instance.noise_type
    if noise_type == 'deterministic':
        logger.debug(
            'Creating MLE objective function with sigma fixed to {0}'.format(xi)
        )
        obj = make_objective_fixed_xi(x, y, xi, jitter)
        theta0 = get_log_theta_init(x, y, infer_noise=False)
    else:  # noise type == 'infer'
        logger.debug('Creating MLE objective function.')
        obj = make_objective_infer_xi(x, y, jitter)
        theta0 = get_log_theta_init(x, y, infer_noise=True)
    ret = optimise_objective(obj, theta0)
    return mle_glue(interface_instance, ret)
   

def make_objective_infer_xi(x, y, xi, jitter):
    _objective = make_mle_objective(x, y, jitter)
    return _objective
   

def make_objective_fixed_xi(x, y, xi, jitter):
    """ Objective function for when the noise standard deviation is fixed."""
    _objective = make_mle_objective(x, y, jitter)

    def objective(theta_):
        theta = np.insert(theta_, 1, xi)
        return _objective(theta)
    return objective


def optimise_objective(objective, theta0):
    """ Interface to the optimiser."""
    ret = optimize.minimize(
        objective,
        theta0,
        method='bfgs',
    )
    return ret


def get_log_theta_init(x, y, infer_noise=False):
    """ Get initial log_theta for the optimiser."""
    amp_init = y.std()
    ls_init = np.std(x, axis=0) / 3
    if infer_noise:
        sigma_init = y.std() / 10
        theta = np.hstack((amp_init, sigma_init, ls_init))
    else:
        theta = np.hstack((amp_init, ls_init))
    return np.log(theta)


def mle_glue(interface_instance, ret):
    """ Extract results and put hyperparameters into a dictionary. """
    if not ret['success']:
        raise RuntimeError(ret['message'])
    theta = np.exp(ret['x'])
    noise_type, xi = interface_instance.noise_type
    if noise_type == 'infer':
        hyperparameters = {
            'amp': np.array([theta[0]], order='F'),
            'xi': np.array([theta[1]], order='F'),
            'ls': np.atleast_2d(theta[2:])
        }
    else:
        hyperparameters = {
            'amp': np.array([theta[0]], order='F'),
            'xi': np.array([xi], order='F', dtype=np.float64),
            'ls': np.atleast_2d(theta[1:])
        }
    info = {
        'method': 'mle',
        'log_lik/N': ret['fun'],
        'message': ret['message'],
        'iterations': ret['nit']
    }
    return hyperparameters, info
