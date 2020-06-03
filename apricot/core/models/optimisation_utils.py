"""

 Docstring


This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import Optional, Mapping, Any, Tuple
import numpy as np  # type: ignore
from apricot.core import sampling
from apricot.core.models import type_aliases as ta


def get_bounds_grid(
        index_dimension: int,
        amp_lower: float,
        amp_upper: float,
        sigma_lower: float,
        sigma_upper: float,
        ls_lower: float,
        ls_upper: float,
        infer_noise: bool
) -> ta.Bounds:
    """ Create dimensionwise bounds for the hyperparameter grid. """
    # pylint: disable=too-many-arguments
    bounds = [(amp_lower, amp_upper)]
    if infer_noise:
        bounds += [(sigma_lower, sigma_upper)]
    bounds += [(ls_lower, ls_upper)] * index_dimension
    return bounds


def get_theta_grid(
        x_data: np.ndarray,
        y_data: np.ndarray,
        infer_noise: bool = False,
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
        seed: Optional[int] = None
) -> np.ndarray:
    """ Construct the initial hyperparameter grid. """
    # pylint: disable=too-many-arguments, too-many-locals
    index_dimension = x_data.shape[1]
    if infer_noise:
        grid_dimension = index_dimension + 2
    else:
        grid_dimension = index_dimension + 1

    if sigma_upper is None:
        sigma_upper = y_data.std() / 10

    if bounds is None:
        bounds = get_bounds_grid(
            index_dimension,
            amp_lower,
            amp_upper,
            sigma_lower,
            sigma_upper,
            ls_lower,
            ls_upper,
            infer_noise
        )
    raw_grid = exp_grid(
        sampling.sample_hypercube(
            grid_size,
            grid_dimension,
            method=grid_method,
            seed=seed,
            options=grid_options
        )
    )
    theta_grid = np.empty_like(raw_grid, dtype=np.float64)
    for i, _b in enumerate(bounds):
        theta_grid[:, i] = scale_vector(raw_grid[:, i], *_b)
    return theta_grid


def theta_grid_search(
        likelihood_func: ta.InternalObjective,
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
        (grid_size, grid_dimension) array of hyperparameter samples for which
        likelihood_func should be queried.

    Returns
    -------
    theta0: ndarray
        (grid_dimension) array. The row of theta_grid at which likelihood
        function is a minimum.
    likf_theta0: float
        The value of likelihood_func at theta0.
    """
    grid_size = theta_grid.shape[0]
    f_grid = np.empty(grid_size, dtype=np.float64)
    for i in range(grid_size):
        f_grid[i] = likelihood_func(theta_grid[i, :])
    argmin = f_grid.argmin()
    return theta_grid[argmin, :], f_grid[argmin]


def exp_grid(grid: np.ndarray) -> np.ndarray:
    """Take the exponent of grid then scale the points between 0 and 1"""
    return (np.exp(grid) - 1) / (np.e - 1)


def scale_vector(
        vector: np.ndarray,
        lower: float,
        upper: float
) -> np.ndarray:
    """ Scale vector on [0, 1] to be between [lower, upper]"""
    return (vector * (upper - lower)) + lower
