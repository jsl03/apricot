import typing
import functools

import numpy as np

from scipy import special
from scipy import optimize

from apricot.core import utils

def ls_inv_gamma_prior(
        x : np.ndarray,
        options : typing.Optional[typing.Union[typing.List[str], str]] = None,
        seed : typing.Optional[int] = None,
):
    """ Inverse Gamma lengthscale hyperprior

    Calculates the parameters of an inverse gamma distribution, ls_alpha and
    ls_beta given, data, x, such that a specified amount of probability mass
    lies below the minimum spacing of the data, and above the support of the
    function in that dimension (assumed to be [0,1]).

    A hard lower limit for the minimum spacing is set at 0.05.

    Parameters
    ----------
    x : ndarray
        (n,d) array of sample points.
    options : {None, str, list of str} optional
        Fit options to fine-tune the behaviour of the prior distribution. If
        a single string is supplied, the same options will be used for each
        input dimension. If a length d list of strings is provided, options
        corresponding to the respective element of the list will be used for
        each dimension of the index.
    seed : {None, int}
        Seed for numpy's random state. If None, an arbitrary seed will be used.
        Default = None.
   
    Returns
    -------
    ls_alpha : ndarray
        (d,) array of alpha parameters for the inverse gamma hyperprior.
    ls_beta : ndarray
        (d,) array of beta parameters for the inverse gamma hyperprior.

    Notes
    -----
    More options to be added. 
    """
    d = x.shape[1]
    delta_min = min_spacing(x)
    options_formatted = format_options(options, d)
    ls_alpha = np.empty(d)
    ls_beta = np.empty(d)
    for dim, option in enumerate(options_formatted):
        ls_alpha[dim], ls_beta[dim] = solve_inv_gamma(*parse_option(delta_min[dim], option), seed=seed)
    return ls_alpha, ls_beta

def min_spacing(x : np.ndarray):
    """Get the minimum spacing between values for each column of x

    Parameters
    ----------
    x : ndarray
        (n, d) array of n points in d dimensions

    Returns
    -------
    delta_min : ndarray
        (d,) array containing the minimum spacing of the data in each of the
        d dimensions in x

    """
    n,d = x.shape
    delta_min = np.empty(d)
    for i in range(d):
        delta = np.abs(np.subtract.outer(x[:,i], x[:,i]))
        delta_masked = np.ma.masked_array(delta, mask=np.eye(n))
        delta_min[i] = np.min(delta_masked)
    return delta_min

def inv_gamma_cdf(x : float, alpha : float, beta : float):
    """ Inverse gamma distribution CDF. """
    if x <= 0:
        return 0.0
    else:
        return special.gammaincc(alpha, beta/x)

def inv_gamma_pdf(x : float, alpha : float, beta : float):
    """ Inverse gamma distribution PDF. """
    if x <= 0:
        return 0.0
    else:
        y=((beta**alpha)/special.gamma(alpha))*x**(-(alpha+1.))*np.exp(-beta*(1./x))
        return y

def inv_gamma_tail(
        lower : float,
        upper : float,
        lower_tol : float,
        upper_tol : float,
        theta : typing.Tuple[float, float],
):
    """ Inverse gamma tail probabilities in excess of tolerances.

    Returns the probability mass of an inverse gamma distribution parametrised
    by theta that is below 'lower' and above 'upper', in excess of specified
    tolerances 'lower_tol' and 'upper_tol', respectively.

    Parameters
    ----------
    lower : float
        Lower bound (nominally, the smallest separation between points in that
        dimension).
    upper : float
        Upper bound (nominally, the support of the function in that dimension,
        assumed by apricot to be 1).
    lower_tol : float
        The algorithm will try to ensure that lower_tol probability mass is <
        lower.
    upper_tol : float
        The algorithm will try to ensure that upper_tol probability mass is >
        upper.
    theta : ndarray
        Tuple of inverse gamma distribution parameters corresponding to
        (alpha, beta).

    Returns
    -------
    lower : float
        Total probability mass less than lower, minus lower_tol
    upper : float
        Total probability mass greater than upper, minus upper_tol
    """
    ls_alpha = theta[0]
    ls_beta = theta[1]
    lower = inv_gamma_cdf(lower, ls_alpha, ls_beta) - lower_tol
    upper = (1.0 - inv_gamma_cdf(upper, ls_alpha, ls_beta)) - upper_tol
    return lower, upper

def create_objective(
        lower : float,
        upper : float,
        lower_tol : float,
        upper_tol : float
):
    """ Objective function for solve_inv_gamma.

    Parameters
    ----------
    lower : float
        Lower bound (nominally, the smallest separation between points in that
        dimension).
    upper : float
        Upper bound (nominally, the support of the function in that dimension,
        assumed by apricot to be 1).
    lower_tol : float
        The algorithm will try to ensure that lower_tol probability mass is <
        lower.
    upper_tol : float
        The algorithm will try to ensure that upper_tol probability mass is >
        upper.

    Returns
    -------
    objective : callable
        Function: objective((alpha, beta)) -> (objective_1, objective_2)
    """
    return functools.partial(inv_gamma_tail, lower, upper, lower_tol, upper_tol)

def solve_inv_gamma(
        lb : float,
        ub : float,
        lb_tol : float,
        ub_tol : float,
        gridsize : int = 10000,
        max_attempts : int = 3,
        seed : typing.Optional[int] = None,
):
    """ Solve system of equations to find appropriate inverse gamma parameters.

    Aims to identify parameters alpha and beta such that:
    * A total of lb_tol probability mass lies < lb
    * A total of ub_tol probability mass lies > ub

    Given an inverse gamma distribution parametrised by alpha and beta.

    Scipy's root finding module scipy.optimize.root is used to solve the
    above system of equations, following a preliminary grid search used to
    identify a suitable starting point.

    Parameters
    ----------
    lb : float
        Lower bound.
    ub : float
        Upper bound.
    lb_tol : float
        Lower bound tolerance.
    ub_tol : float
        Upper bound tolerance.
    gridsize : int, optional
        Size of grid used for preliminary grid search.
    max_attempts : int, optional
        Maximum number of attempts permitted.
    seed : {None, int32}
        Seed for numpy's random state. If None, an arbitrary random seed will
        be used. Default = None.

    Returns
    -------
    alpha : float
        Inverse gamma parameter alpha.
    beta : float
        Inverse gamma parameter beta.
    """
    set_seed(seed)
    if lb > ub:
        raise ValueError('Lower bound cannot be greater than upper bound.')
    obj = create_objective(lb, ub, lb_tol, ub_tol)
    attempts = 1
    converged = False
    scales = np.array([10, 10])
    obj_grid = np.empty((gridsize, 2))
    while not converged:
        theta_grid = np.random.random((gridsize, 2))*scales
        # objective function is not vectorised, so run in loop...
        for i in range(gridsize):
            obj_grid[i,:] = obj(theta_grid[i,:])
        obj_grid_norm = np.sqrt(np.sum(obj_grid**2, axis=1))
        theta0 = theta_grid[obj_grid_norm.argmin(), :]
        theta_sol= optimize.root(obj, theta0)
        converged = theta_sol['success']
        if attempts > max_attempts:
            raise RuntimeError('Maximum number of attempts exceeded without convergence.')
    return theta_sol['x'][0], theta_sol['x'][1]

def format_options(
        options : typing.Optional[typing.Union[typing.List[str], str]],
        d: int
):
    """ Ensure 'options' is a list of length d.

    Options must be present for each input dimension. If only one options
    string is provided, it is "cloned" d times. Lists are unmodified.
    """
    if options is None:
        return [None for _ in range(d)]
    if type(options) is str:
        return [options for _ in range(d)]
    else:
        return options

def parse_option(delta_min : float, option : str):
    """ Parse options.

    Parse requested option and designate lb, ub, lb_tol and ub_tol.

    Defaults to 'nonlinear' if option is None.
    """
    if option is None:
        option = 'nonlinear'
    if option.lower() == 'nonlinear':
        lb = max(delta_min, 0.05)
        ub = 1
        lb_tol = 0.01
        ub_tol = 0.01
    elif option.lower() == 'linear':
        lb = 1
        ub = 5.
        lb_tol = 0.05
        ub_tol = 0.2
    else:
        raise NotImplementedError('{}'.format(option))
    return lb, ub, lb_tol, ub_tol
