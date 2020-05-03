# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import (Callable, Tuple, Optional, Dict, Any, Mapping, cast)
import numpy as np  # type: ignore
from scipy import optimize  # type: ignore
from apricot.core.gp_internal import NLMLEqKernel  # type: ignore
from apricot.core.sampling import sample_hypercube
from apricot.core.logger import get_logger
from apricot.core.models import type_aliases as ta


# satisfy forward type checking
if False:  # pylint: disable=using-constant-test
    import apricot  # pylint: disable=unused-import


LOGGER = get_logger()


def run_mle(
        interface_instance: 'apricot.core.models.interface.Interface',
        x: np.ndarray,
        y: np.ndarray,
        jitter: float = 1e-10,
        bounds: Optional[ta.Bounds] = None,
        ls_lower: float = 0.05,
        ls_upper: float = 1.0,
        xi_lower: float = 0.0,
        xi_upper: Optional[float] = None,
        amp_lower: float = 0.0,
        amp_upper: float = 1.0,
        grid_size: int = 5000,
        grid_method: str = 'sobol',
        grid_options: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        callback: Optional[ta.CallbackFunction] = None,
):
    """ Maximum Log Marginal Likelihood Estimator

    Parameters
    ----------
    interface_instance: intstance of apricot.core.models.interface.Interface
        Interface to the model who's hyperparameters should be identified.
    x: ndarray
        (n, d) array of n sample points in d dimensions.
    y: ndarray
        (n) array of n responses corresponding to the rows of x.
    jitter: float, optional
        Magnitude of stability jitter to be added to the leading diagonal of
        the sample covariance matrix.
    bounds: List of tuple, optional
        List of (lower, upper) bounds for the preliminary grid search for each
        hyperparameter in the following order: signal amplitude (marginal
        standard deviation), (optionally) noise variance (if present),
        anisotropic lengthscales. For fixed noise models, this is:
        [amp, ls_1, ..., ls_d], and if inferring the noise standard deviation,
        this is [amp, xi, ls_1, ..., ls_d]. If not provided, the grid search
        bounds will be set to "sensible" defaults (see below). Default = None.
    ls_lower: float, optional
        Grid search lower bound for anisotropic lengthscales. Default = 0.05.
    ls_upper: float, optional
        Grid search upper bound for anisotropic lengthscales. Default = 1.
    xi_lower: float, optional
        Grid search lower bound for noise standard deviation. Default = 0
    xi_upper: {None, float}, optional
        Grid search upper bound for noise standard deviation. If None and
        infer_noise is True, xi_upper is equal to 1/10 of the
        sample standard deviation for function observations, that is:
        xi_upper = np.std(y) / 10. Default = None.
    amp_lower: float, optional
        Grid search lower bound for marginal standard deviation. Default = 0.
    amp_upper: float, optional
        Grid search upper bound for marginal standard deviation. Default = 1.
    grid_size: int, optional
        Number of points to use for the preliminary grid search. Default =
        5000. For large sample sample sizes, this can be reduced.
    grid_method: str, optional
        String specifying which method to use to generate the grid points.
        Valid options are compatible with apricot.sample_hypercube. Default =
        'sobol'.
    grid_options: dict, optional
        Additional options to pass to the chosen grid generating method.
        Default = None.
    seed: int32, optional
        Random seed. If not provided, one will be generated. Default = None.
    callback: ndarray -> Any
        Callback function. Receives the parameter vector queried by the
        optimisation algorithm at each iteration. Used primarily for debugging.
    """
    noise_type, xi = interface_instance.noise_type
    kernel_type = interface_instance.kernel_type
    if kernel_type != 'eq':
        msg = 'MLE is currently only implemented for the EQ kernel'
        raise ValueError(msg)
    grid_opts_dict: Dict[str, Any] = {
        'bounds': bounds,
        'ls_lower': ls_lower,
        'ls_upper': ls_upper,
        'xi_lower': xi_lower,
        'xi_upper': xi_upper,
        'amp_lower': amp_lower,
        'amp_upper': amp_upper,
        'grid_size': grid_size,
        'grid_method': grid_method,
        'grid_options': grid_options,
        'seed': seed
    }
    if noise_type == 'deterministic':
        xi = cast(float, xi)  # let mypy know xi is always a float here
        obj, obj_jac = make_objective_fixed_xi(x, y, xi, jitter)
        grid_opts_dict['infer_noise'] = False
        opts = cast(Mapping[str, Any], grid_opts_dict)
        theta_grid = get_theta_grid(x, y, **opts)
    elif noise_type == 'infer':
        obj, obj_jac = make_objective_infer_xi(x, y, jitter)
        grid_opts_dict['infer_noise'] = True
        opts = cast(Mapping[str, Any], grid_opts_dict)
        theta_grid = get_theta_grid(x, y, **opts)
    else:
        raise RuntimeError('Unknown noise type.')
    theta0, ftheta0 = theta_grid_search(obj, theta_grid)
    LOGGER.debug(theta0)
    ret = run_optimiser(obj_jac, theta0, callback)
    return mle_glue(interface_instance, ret)


def run_optimiser(
        objective: ta.NLMLJac,
        theta0: np.ndarray,
        callback: Optional[ta.CallbackFunction] = None,
) -> Dict[str, Any]:
    """ Interface to the optimiser.

    Parameters
    ----------
    objective: ndarray -> (float, ndarray)
        Objective function. Accepts a vector, theta, containing the
        hyperparameters [amp, <xi>, ls_1, ..., ls_d], and returns a tuple
        containing the value of the negative log marginal likelihood and its
        partial derivatives with respect to theta. Xi is added automatically
        for NLML functions created by apricot which exhibit fixed noise.
    theta0: ndarray
        Initial values of the hyperparameters from which to start the
        optimiser.
    callback: ndarray -> Any
        Optional callback function. Primarily used for debugging.
        Default = None.

    Returns
    -------
    opt_result: OptimizeResult
        A raw OptimizeResult object returned directly from
        <scipy.optimize.minimize>.

    Notes
    -----
    Add support for different optimisers, pass keyword arguments, etc.

    """
    opt_result = optimize.minimize(
        objective,
        theta0,
        jac=True,
        method='bfgs',
        callback=callback
    )
    return opt_result


def get_bounds_grid(  # pylint: disable=C0103, R0913
        d_index: int,
        amp_lower: float,
        amp_upper: float,
        xi_lower: float,
        xi_upper: float,
        ls_lower: float,
        ls_upper: float,
        infer_noise: bool
) -> ta.Bounds:
    """ Create dimensionwise bounds for the hyperparameter grid. """
    bounds = [(amp_lower, amp_upper)]
    if infer_noise:
        bounds += [(xi_lower, xi_upper)]
    bounds += [(ls_lower, ls_upper)] * d_index
    return bounds


def get_theta_grid(  # pylint: disable=C0103, R0913, R0914
        x: np.ndarray,
        y: np.ndarray,
        infer_noise: bool = False,
        bounds: Optional[ta.Bounds] = None,
        ls_lower: float = 0.05,
        ls_upper: float = 1.0,
        xi_lower: float = 0.0,
        xi_upper: Optional[float] = None,
        amp_lower: float = 0.0,
        amp_upper: float = 1.0,
        grid_size: int = 5000,
        grid_method: str = 'sobol',
        grid_options: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None
) -> np.ndarray:
    """ Construct the initial hyperparameter grid. """
    d_index = x.shape[1]
    if infer_noise:
        d_grid = d_index + 2
    if xi_upper is None:
        xi_upper = y.std() / 10
    else:
        d_grid = d_index + 1
    if bounds is None:
        bounds = get_bounds_grid(
            d_index,
            amp_lower,
            amp_upper,
            xi_lower,
            xi_upper,
            ls_lower,
            ls_upper,
            infer_noise
        )
    raw_grid = _exp_grid(
        sample_hypercube(
            grid_size,
            d_grid,
            method=grid_method,
            seed=seed,
            options=grid_options
        )
    )
    theta_grid = np.empty_like(raw_grid, dtype=np.float64)
    for i, _b in enumerate(bounds):
        theta_grid[:, i] = _scale_vector(raw_grid[:, i], *_b)
    return theta_grid


def theta_grid_search(
        likelihood_func: ta.NLML,
        theta_grid: np.ndarray
) -> Tuple[np.ndarray, float]:
    """ Hyperparameter grid Search

    Evaluate likelihood_func over the rows of theta_grid and return the row
    corresponding to the minimum, along with the value of likelihood_func
    corresponding to that row.

    Parameters
    ----------
    likelihood_func: ndarray -> float
        Negative log marginal likelihood function to evaluate over theta_grid.
    theta_grid: ndarray
        (n, d_grid) array of hyperparameter samples for which likelihood_func
        should be queried.

    Returns
    -------
    theta0: ndarray
        (d_grid) array. The row of theta_grid at which likelihood function is a
        minimum.
    likf_theta0: float
        The value of likelihood_func at theta0.
    """
    grid_size = theta_grid.shape[0]
    f_grid = np.empty(grid_size, dtype=np.float64)
    for i in range(grid_size):
        f_grid[i] = likelihood_func(theta_grid[i, :])
    argmin = f_grid.argmin()
    return theta_grid[argmin, :], f_grid[argmin]


def make_objective_infer_xi(  # pylint: disable=C0103,
        x: np.ndarray,
        y: np.ndarray,
        jitter: float = 1e-10
) -> Tuple[ta.NLML, ta.NLMLJac]:
    """ Create objective function with variable noise standard deviation.

    Creates a negative log marginal likelihood (NLML) function and a function
    returning its derivatives for the exponentiated quadratic (EQ) kernel,
    given sample points x, sample responses y, and stability jitter of
    magnitude jitter.

    Both the objective function and the function returning its derivatives
    accept a length (d+2) vector theta, containing the following parameters in
    this specific order:
        * marginal standard deviation, "amp".
        * Additive Gaussian noise standard deviation, "xi".
        * A total of d anisotropic lengthscales, "ls".

    i.e., theta should be a vector of [amp, xi, ls_1, ..., ls_d].

    Parameters
    ----------
    x: ndarray
        (n, d) array of n sample points in d dimensions.
    y: (n) array
        (n) array of sample responses, corresponding to the rows of x
    jitter: float, optional
        Stability jitter of the specified magnitude will be added to
        the leading diagonal of the sample covariance matrix to ensure
        it remains positive semi defininite. Default = 1e-10.

    Returns
    -------
    objective: [ndarray] -> float
        The negative log marginal likelihood of a GP with an EQ kernel given
        data (x, y) and the hyperparameters described by theta.
    objective_jac: [ndarray] -> float, ndarray
        The negative log marginal likelihood of a GP with an EQ kernel given
        data (x, y) and the hyperparameters described by theta, in addition to
        a length (d+2) vector of derivatives of the negative log marginal
        likelihood with respect to theta.
    """
    LOGGER.debug("Creating variable noise NLML function.")
    objective_ = NLMLEqKernel(x, y, jitter)

    def objective(theta: np.ndarray) -> float:
        return objective_(theta)

    def objective_jac(theta: np.ndarray) -> Tuple[np.ndarray, float]:
        if any(theta <= 0):
            return np.inf, np.full_like(theta, np.inf)
        return objective_.jac(theta)

    return objective, objective_jac


def make_objective_fixed_xi(  # pylint: disable=C0103,
        x: np.ndarray,
        y: np.ndarray,
        xi: float,
        jitter: float = 1e-10
) -> Tuple[ta.NLML, ta.NLMLJac]:
    """ Create objective function with fixed noise standard deviation.

    Creates a negative log marginal likelihood (NLML) function and a function
    returning its derivatives for the exponentiated quadratic (EQ) kernel,
    given sample points x, sample responses y, additive Gaussian noise of
    standard deviation xi, and stability jitter of magnitude jitter.

    Both the objective function and the function returning its derivatives
    accept a length (d+1) vector theta, containing the following parameters in
    this specific order:
        * marginal standard deviation, "amp".
        * A total of d anisotropic lengthscales, "ls".

    i.e., theta should be a vector of [amp, ls_1, ..., ls_d].

    The requested value of xi is assigned automatically.

    Parameters
    ----------
    x: ndarray
        (n, d) array of n sample points in d dimensions.
    y: (n) array
        (n) array of sample responses, corresponding to the rows of x.
    xi: float >= 0
        Standard deviation of Gaussian noise to be added to the leading
        diagonal of the sample covariance matrix. Is permitted to be 0, but
        may not be negative.
    jitter: float, optional
        Stability jitter of the specified magnitude will be added to
        the leading diagonal of the sample covariance matrix to ensure
        it remains positive semi defininite. Default = 1e-10.

    Returns
    -------
    objective: [ndarray] -> float
        The negative log marginal likelihood of a GP with an EQ kernel given
        data (x, y) and the hyperparameters described by theta.

    objective_jac: [ndarray] -> float, ndarray
        The negative log marginal likelihood of a GP with an EQ kernel given
        data (x, y) and the hyperparameters described by theta, in addition to
        a length (d+1) vector of derivatives of the negative log marginal
        likelihood with respect to theta.
    """
    LOGGER.debug("Creating NLML function with xi=%s", xi)
    objective_ = NLMLEqKernel(x, y, jitter)

    def objective(theta_partial):
        theta = np.insert(theta_partial, 1, xi)
        return objective_(theta)

    def objective_jac(theta_partial):
        if any(theta_partial.ravel() <= 0):
            return np.inf, np.full_like(theta_partial, np.inf)
        theta = np.insert(theta_partial, 1, xi)
        # need to remove derivative in position 1 since xi is fixed
        return fixed_xi_output_helper(*objective_.jac(theta))

    return objective, objective_jac


def mle_glue(
        interface_instance,
        ret: dict,
) -> Tuple[ta.Hyperparameters, Dict[str, Any]]:
    """ Extract results and put hyperparameters into a dictionary. """
    # TODO: protocol for dealing with failures (usually precision loss)
    #  if not ret['success']:
    #  raise RuntimeError(ret['message'])
    theta = ret['x']
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


def fixed_xi_output_helper(
        loglik: float,
        jac: np.ndarray
) -> Tuple[float, np.ndarray]:
    """ Delete derivative in position 1 from objective.jac """
    return loglik, np.delete(jac, 1)


def _exp_grid(grid: np.ndarray) -> np.ndarray:
    """Take the exponent of grid then scale the points between 0 and 1"""
    return (np.exp(grid) - 1) / (np.e - 1)


def _scale_vector(
        vector: np.ndarray,
        lower: float,
        upper: float
) -> np.ndarray:
    return (vector * (upper - lower)) + lower
