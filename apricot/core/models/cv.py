"""
Analytical leave-one-out cross validation score optimisation.

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import Tuple, Optional, Dict, Any, Mapping, cast
import numpy as np  # type: ignore
from scipy import optimize  # type: ignore
from apricot.core.gp_internal import CVEqKernel  # type: ignore
from apricot.core.logger import get_logger
from apricot.core.models import type_aliases as ta
from apricot.core.models import optimisation_utils as ou


# satisfy forward type checking
if False:  # pylint: disable=using-constant-test
    import apricot  # pylint: disable=unused-import


LOGGER = get_logger()


def run_cv(
        interface_instance: 'apricot.core.models.interface.Interface',
        x_data: np.ndarray,
        y_data: np.ndarray,
        jitter: float = 1e-10,
        bounds: Optional[ta.Bounds] = None,
        ls_lower: float = 0.05,
        ls_upper: float = 1.0,
        sigma_lower: float = 0.0,
        sigma_upper: Optional[float] = None,
        amp_lower: float = 0.0,
        amp_upper: float = 1.0,
        grid_size: int = 5000,
        grid_method: str = 'sobol',
        grid_options: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        callback: Optional[ta.CallbackFunction] = None,
):
    """ Leave-One-Out Cross Validation

    Parameters
    ----------
    interface_instance: intstance of apricot.core.models.interface.Interface
        Interface to the model who's hyperparameters should be identified.
    x_data: ndarray
        (n, d) array of n sample points in d dimensions.
    y_data: ndarray
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
    sigma_lower: float, optional
        Grid search lower bound for noise standard deviation. Default = 0
    sigma_upper: {None, float}, optional
        Grid search upper bound for noise standard deviation. If None and
        infer_noise is True, sigma_upper is equal to 1/10 of the
        sample standard deviation for function observations, that is:
        sigma_upper = np.std(y) / 10. Default = None.
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
    # pylint: disable=too-many-arguments, too-many-locals
    noise_type, sigma = interface_instance.noise_type
    kernel_type = interface_instance.kernel_type
    if kernel_type != 'eq':
        msg = 'CV is currently only implemented for the EQ kernel.'
        raise ValueError(msg)
    grid_opts_dict: Dict[str, Any] = {
        'bounds': bounds,
        'ls_lower': ls_lower,
        'ls_upper': ls_upper,
        'sigma_lower': sigma_lower,
        'sigma_upper': sigma_upper,
        'amp_lower': amp_lower,
        'amp_upper': amp_upper,
        'grid_size': grid_size,
        'grid_method': grid_method,
        'grid_options': grid_options,
        'seed': seed
    }
    if noise_type == 'deterministic':
        sigma = cast(float, sigma)  # sigma is always a float here
        obj = make_objective_fixed_noise(
            x_data,
            y_data,
            sigma,
            jitter
        )
        grid_opts_dict['infer_noise'] = False
        opts = cast(Mapping[str, Any], grid_opts_dict)
        theta_grid = ou.get_theta_grid(x_data, y_data, **opts)
    elif noise_type == 'infer':
        obj = make_objective_infer_noise(
            x_data,
            y_data,
            jitter
        )
        grid_opts_dict['infer_noise'] = True
        opts = cast(Mapping[str, Any], grid_opts_dict)
        theta_grid = ou.get_theta_grid(x_data, y_data, **opts)
    else:
        raise RuntimeError('Unknown noise type.')
    theta0, ftheta0 = ou.theta_grid_search(obj, theta_grid)
    LOGGER.debug(
        'theta0: %(theta0)s CV: %(f)s',
        {'theta0': theta0, 'f': ftheta0}
    )
    ret = run_optimiser(obj, theta0, callback)
    return cv_glue(interface_instance, ret)


def run_optimiser(
        objective,
        theta0: np.ndarray,
        callback: Optional[ta.CallbackFunction] = None,
) -> Dict[str, Any]:
    """ Interface to the optimiser.

    Parameters
    ----------
    objective: ndarray -> (float, ndarray)
        Objective function. Accepts a vector, theta, containing the
        hyperparameters [amp, <sigma>, ls_1, ..., ls_d], and returns CV.
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
        jac=None,
        method='bfgs',
        callback=callback
    )
    return opt_result


def make_objective_infer_noise(
        x_data: np.ndarray,
        y_data: np.ndarray,
        jitter: float = 1e-10
) -> ta.InternalObjective:
    """ Create objective function with variable noise standard deviation.

    Creates a leave-one-out cross validation (CV) function for the
    exponentiated quadratic (EQ) kernel, given sample points x, sample
    responses y, and stability jitter of magnitude jitter.

    The objective function  accepts a length (d+2) vector theta, containing the
    following parameters in this specific order:
        * marginal standard deviation, "amp".
        * Additive Gaussian noise standard deviation, "sigma".
        * A total of d anisotropic lengthscales, "ls".

    i.e., theta should be a vector of [amp, sigma, ls_1, ..., ls_d].

    Parameters
    ----------
    x_data: ndarray
        (n, d) array of n sample points in d dimensions.
    y_data: (n) array
        (n) array of sample responses, corresponding to the rows of x
    jitter: float, optional
        Stability jitter of the specified magnitude will be added to
        the leading diagonal of the sample covariance matrix to ensure
        it remains positive semi defininite. Default = 1e-10.

    Returns
    -------
    objective: [ndarray] -> float
        The negative mean Leave-One-Out predictive density of a GP with an
        EQ kernel given data (x, y) and the hyperparameters described by
        theta.
    """
    LOGGER.debug("Creating variable noise LOO-CV function.")
    objective_ = CVEqKernel(x_data, y_data, jitter)

    def objective(theta: np.ndarray) -> float:
        return objective_(theta)

    return objective


def make_objective_fixed_noise(
        x_data: np.ndarray,
        y_data: np.ndarray,
        sigma: float,
        jitter: float = 1e-10
) -> ta.InternalObjective:
    """ Create objective function with fixed noise standard deviation.

    Creates a leave-one-out cross validation function for the exponentiated
    quadratic (EQ) kernel, given sample points x, sample responses y,
    additive Gaussian noise of standard deviation sigma, and stability jitter
    of magnitude jitter.

    The objective function accepts a length (d+1) vector, theta, containing
    the following parameters in this specific order:
        * marginal standard deviation, "amp".
        * A total of d anisotropic lengthscales, "ls".

    i.e., theta should be a vector of [amp, ls_1, ..., ls_d].

    The requested value of sigma is assigned automatically.

    Parameters
    ----------
    x_data: ndarray
        (n, d) array of n sample points in d dimensions.
    y_data: (n) array
        (n) array of sample responses, corresponding to the rows of x.
    sigma: float >= 0
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
        The negative mean Leave-One-Out predictive density of a GP with an
        EQ kernel given data (x, y) and the hyperparameters described by
        theta.
    """
    LOGGER.debug("Creating LOO-CV function with sigma=%s", sigma)
    objective_ = CVEqKernel(x_data, y_data, jitter)

    def objective(theta_partial):
        theta = np.insert(theta_partial, 1, sigma)
        return objective_(theta)

    return objective


def cv_glue(
        interface_instance,
        ret: dict,
) -> Tuple[ta.Hyperparameters, Dict[str, Any]]:
    """ Extract results and put hyperparameters into a dictionary. """
    # TODO: protocol for dealing with failures (usually precision loss)
    #  if not ret['success']:
    #  raise RuntimeError(ret['message'])
    theta = ret['x']
    noise_type, sigma = interface_instance.noise_type
    if noise_type == 'infer':
        hyperparameters = {
            'amp': np.array([theta[0]], order='F'),
            'sigma': np.array([theta[1]], order='F'),
            'ls': np.atleast_2d(theta[2:])
        }
    else:
        hyperparameters = {
            'amp': np.array([theta[0]], order='F'),
            'sigma': np.array([sigma], order='F', dtype=np.float64),
            'ls': np.atleast_2d(theta[1:])
        }
    info = {
        'method': 'CV',
        'Mean density': ret['fun'],
        'message': ret['message'],
        'iterations': ret['nit']
    }
    return hyperparameters, info
