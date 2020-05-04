# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import Optional, Tuple, Sequence
import numpy as np  # type: ignore
from scipy import special  # type: ignore
from scipy import optimize  # type: ignore
from apricot.core import utils
from apricot.core.models import type_aliases as ta


def ls_inv_gamma_prior(
        x_data: np.ndarray,
        ls_options: Optional[ta.LsPriorOptions] = None,
        gridsize: int = 10000,
        max_attempts: int = 3,
        seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Inverse Gamma lengthscale hyperprior.

    Calculates the parameters of an inverse gamma distribution, ls_alpha and
    ls_beta given, data, x, such that a specified amount of probability mass
    lies below the minimum spacing of the data, and above the support of the
    function in that dimension (assumed to be [0,1]).

    A hard lower limit for the minimum spacing is set at 0.05.

    Parameters
    ----------
    x_data: ndarray
        (n, d) array of sample points.
    options: {None, str, list of str} optional
        Fit options to fine-tune the behaviour of the prior distribution. If
        a single string is supplied, the same options will be used for each
        input dimension. If a length d list of strings is provided, options
        corresponding to the respective element of the list will be used for
        each dimension of the index:
        * None : defaults to 'nonlinear'.
        * 'nonlinear' : 0.01 prior mass mass below minimum spacing and above 1
        * 'linear' : increases the amount of probability mass permissible > 1.
    seed: {None, int}
        Seed for numpy's random state. If None, an arbitrary seed will be used.
        Default = None.

    Returns
    -------
    ls_alpha: ndarray
        (d,) array of alpha parameters for the inverse gamma hyperprior.
    ls_beta: ndarray
        (d,) array of beta parameters for the inverse gamma hyperprior.

    Notes
    -----
    More options to be added.
    """
    index_dimension = x_data.shape[1]
    delta_min = min_spacing(x_data)
    formatted_options = format_options(ls_options, index_dimension)
    ls_alpha = np.empty(index_dimension)
    ls_beta = np.empty(index_dimension)
    for dim, option in enumerate(formatted_options):
        lower_bound, upper_bound, lower_tol, upper_tol = parse_option(
            delta_min[dim],
            option
        )
        ls_alpha[dim], ls_beta[dim] = solve_inv_gamma(
            lower_bound,
            upper_bound,
            lower_tol,
            upper_tol,
            gridsize=gridsize,
            max_attempts=max_attempts,
            seed=seed,
        )
    return ls_alpha, ls_beta


def min_spacing(x_data: np.ndarray) -> np.ndarray:
    """Get the minimum spacing between values for each column of x

    Parameters
    ----------
    x_data: ndarray
        (n, d) array of n points in d dimensions

    Returns
    -------
    delta_min : ndarray
        (d,) array containing the minimum spacing of the data in each of the
        d dimensions in x_data
    """
    sample_size, index_dimensions = x_data.shape
    delta_min = np.empty(index_dimensions)
    for i in range(index_dimensions):
        delta = np.abs(np.subtract.outer(x_data[:, i], x_data[:, i]))
        delta_masked = np.ma.masked_array(delta, mask=np.eye(sample_size))
        delta_min[i] = np.min(delta_masked)
    return delta_min


def inv_gamma_cdf(x: float, alpha: float, beta: float) -> float:
    """ Inverse gamma distribution CDF of x. """
    # pylint: disable=invalid-name
    if x <= 0:
        return 0.0
    return special.gammaincc(alpha, beta/x)  # pylint: disable=no-member


def inv_gamma_pdf(x: float, alpha: float, beta: float) -> float:
    """ Inverse gamma distribution PDF of x. """
    # pylint: disable=invalid-name
    if x <= 0:
        return 0.0
    return (
        ((beta**alpha)/special.gamma(alpha)) *
        x ** (-(alpha + 1.0)) * np.exp(-beta*(1.0 / x))
    )


def inv_gamma_tail(
        lower_bound: float,
        upper_bound: float,
        lower_tol: float,
        upper_tol: float,
        theta: Tuple[float, float],
) -> Tuple[float, float]:
    """ Inverse gamma tail probabilities in excess of tolerances.

    Returns the probability mass of an inverse gamma distribution parametrised
    by theta that is below 'lower' and above 'upper', in excess of specified
    tolerances 'lower_tol' and 'upper_tol', respectively.

    Parameters
    ----------
    lower_bound: float
        Lower bound (nominally, the smallest separation between points in that
        dimension).
    upper_bound: float
        Upper bound (nominally, the support of the function in that dimension,
        assumed by apricot to be 1).
    lower_tol: float
        The algorithm will try to ensure that lower_tol probability mass is <
        lower.
    upper_tol: float
        The algorithm will try to ensure that upper_tol probability mass is >
        upper.
    theta: ndarray
        Tuple of inverse gamma distribution parameters corresponding to
        (alpha, beta).

    Returns
    -------
    p_lower: float
        Total probability mass less than lower, minus lower_tol
    p_upper: float
        Total probability mass greater than upper, minus upper_tol
    """
    ls_alpha = theta[0]
    ls_beta = theta[1]
    p_lower = inv_gamma_cdf(lower_bound, ls_alpha, ls_beta) - lower_tol
    p_upper = (1.0 - inv_gamma_cdf(upper_bound, ls_alpha, ls_beta)) - upper_tol
    return p_lower, p_upper


def create_objective(
        lower_bound: float,
        upper_bound: float,
        lower_tol: float,
        upper_tol: float
) -> ta.LsOptObjective:
    """ Objective function for solve_inv_gamma.

    Parameters
    ----------
    lower_bound: float
        Lower bound (nominally, the smallest separation between points in that
        dimension).
    upper_bound: float
        Upper bound (nominally, the support of the function in that dimension,
        assumed by apricot to be 1).
    lower_tol: float
        The algorithm will try to ensure that lower_tol probability mass is <
        lower.
    upper_tol: float
        The algorithm will try to ensure that upper_tol probability mass is >
        upper.

    Returns
    -------
    objective: callable
        Function: objective((alpha, beta)) -> (objective_1, objective_2)
    """

    def objective(theta: Tuple[float, float]) -> Tuple[float, float]:
        return inv_gamma_tail(
            lower_bound, upper_bound, lower_tol, upper_tol, theta)
    return objective


def solve_inv_gamma(
        lower_bound: float,
        upper_bound: float,
        lower_tol: float,
        upper_tol: float,
        gridsize: int = 10000,
        max_attempts: int = 3,
        seed: Optional[int] = None,
) -> Tuple[float, float]:
    # pylint: disable= too-many-arguments
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
    lower_bound: float
        Lower bound.
    upper_bound: float
        Upper bound.
    lower_tol: float
        Lower bound tolerance.
    upper_tol: float
        Upper bound tolerance.
    gridsize: int, optional
        Size of grid used for preliminary grid search.
    max_attempts: int, optional
        Maximum number of attempts permitted.
    seed: {None, int32}
        Seed for numpy's random state. If None, an arbitrary random seed will
        be used. Default = None.

    Returns
    -------
    alpha: float
        Inverse gamma parameter alpha.
    beta: float
        Inverse gamma parameter beta.
    """
    utils.set_seed(seed)
    if lower_bound >= upper_bound:
        raise ValueError('Lower bound must be smaller than upper bound.')
    obj = create_objective(lower_bound, upper_bound, lower_tol, upper_tol)
    attempts = 1
    converged = False
    scales = np.array([10, 10])
    obj_grid = np.empty((gridsize, 2))
    while not converged:
        theta_grid = np.random.random((gridsize, 2))*scales
        # objective function is not vectorised, so run in loop...
        for i in range(gridsize):
            obj_grid[i, :] = obj(theta_grid[i, :])
        obj_grid_norm = np.sqrt(np.sum(obj_grid**2, axis=1))
        theta0 = theta_grid[obj_grid_norm.argmin(), :]
        theta_sol = optimize.root(obj, theta0)
        converged = theta_sol['success']
        if attempts > max_attempts:
            raise RuntimeError(
                'Maximum number of attempts exceeded without convergence.'
            )
    return theta_sol['x'][0], theta_sol['x'][1]


def format_options(
        ls_options: Optional[ta.LsPriorOptions],
        index_dimension: int
) -> Sequence[Optional[str]]:
    """ Ensure 'options' is a list of length equivalent to the number of index
    dimensions.

    Options must be present for each input dimension. If only one options
    string is provided, it is "cloned" d times. Lists are unmodified.
    """
    if ls_options is None:
        return [None] * index_dimension
    if isinstance(ls_options, str):
        return [ls_options] * index_dimension
    return ls_options


def parse_option(
        delta_min: float,
        option: Optional[str]
) -> Tuple[float, float, float, float]:
    """ Parse options.

    Parse requested option and designate lb, ub, lb_tol and ub_tol. Defaults
    to 'nonlinear' if option is None.

    Returns
    -------
    lower_bound: float
        Lower bound.
    upper_bound: float
        Upper bound.
    lower_tol: float
        Lower bound tolerance.
    upper_tol: float
        Upper bound tolerance.
    """
    if option is None:
        option = 'nonlinear'
    if option.lower() == 'nonlinear':
        lower_bound = max(delta_min, 0.05)
        upper_bound = 1.0
        lower_tol = 0.01
        upper_tol = 0.01
    elif option.lower() == 'linear':
        lower_bound = 1.0
        upper_bound = 5.0
        lower_tol = 0.01
        upper_tol = 0.01
    else:
        raise NotImplementedError('{}'.format(option))
    return lower_bound, upper_bound, lower_tol, upper_tol
