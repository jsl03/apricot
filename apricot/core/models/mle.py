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
            'Creating MLE objective function with fixed sigma = {0}'.format(xi)
        )
        obj, obj_jac = make_objective_fixed_xi(x, y, xi, jitter)
        theta0 = theta_init_stable(x, y, infer_noise=False)
    elif noise_type == 'infer':
        logger.debug('Creating MLE objective function.')
        obj, obj_jac = make_objective_infer_xi(x, y, jitter)
        theta0 = theta_init_stable(x, y, infer_noise=True)
    else:
        raise RuntimeError('Unknown noise type.')
    ret = optimise_objective(obj_jac, theta0)
    return mle_glue(interface_instance, ret)


def make_objective_infer_xi(x, y, jitter):
    objective = make_mle_objective(x, y, jitter)
    return objective.f, objective.f_jac


def format_output(loglik, grad):
    return loglik, np.delete(grad, 1)


def make_objective_fixed_xi(x, y, xi, jitter):
    """ Objective function for when the noise standard deviation is fixed."""
    _objective = make_mle_objective(x, y, jitter)

    def objective(log_theta_):
        log_theta = np.insert(log_theta_, 1, xi)
        return _objective.f_log(log_theta)

    def objective_jac(log_theta_):
        log_theta = np.insert(log_theta_, 1, xi)
        return format_output(*_objective.f_log_jac(log_theta))

    return objective, objective_jac


def optimise_objective(objective, theta0):
    """ Interface to the optimiser."""
    ret = optimize.minimize(
        objective,
        theta0,
        jac=True,
        method='bfgs',
    )
    return ret


def theta_init_stable(x, y, infer_noise=False):
    """ Get initial log_theta for the optimiser."""
    amp_init = y.std()
    ls_init = np.std(x, axis=0) / 3
    if infer_noise:
        sigma_init = y.std() / 10
        theta = np.hstack((amp_init, sigma_init, ls_init))
    else:
        theta = np.hstack((amp_init, ls_init))
    return theta


def mle_glue(interface_instance, ret, clip_ls=1e4):
    """ Extract results and put hyperparameters into a dictionary. """
    if not ret['success']:
        raise RuntimeError(ret['message'])
    theta = np.exp(ret['x'])
    noise_type, xi = interface_instance.noise_type
    if noise_type == 'infer':
        hyperparameters = {
            'amp': np.array([theta[0]], order='F'),
            'xi': np.array([theta[1]], order='F'),
            'ls': np.atleast_2d(np.clip(theta[2:], 0, clip_ls))
        }
    else:
        hyperparameters = {
            'amp': np.array([theta[0]], order='F'),
            'xi': np.array([xi], order='F', dtype=np.float64),
            'ls': np.atleast_2d(np.clip(theta[1:], 0, clip_ls))
        }
    info = {
        'method': 'mle',
        'log_lik/N': ret['fun'],
        'message': ret['message'],
        'iterations': ret['nit']
    }
    return hyperparameters, info
