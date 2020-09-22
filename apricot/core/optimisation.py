"""
Optimisation routines for the Emulator class and associated utilities.

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import Optional, Mapping, Any
import numpy as np  # type: ignore
from scipy.optimize import minimize  # type: ignore
from apricot.core.utils import force_f_array
from apricot.core.models import type_aliases as ta
from apricot.core.sampling import sample_hypercube


def optimise(  # pylint: disable=too-many-arguments
        func: ta.ObjectiveFunction,
        func_and_jac: ta.ObjectiveFunctionJac,
        index_dimension: int,
        x0: Optional[np.ndarray] = None,
        grid: Optional[np.ndarray] = None,
        grid_size: Optional[int] = None,
        grid_method: str = 'lhs',
        grid_options: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
) -> dict:
    """ Generic numerical optimisation routine using SLSQP.

    Generic numerical optimisation strategy consisting of a preliminary grid
    search and a gradient based optimisation algorithm implemented in scipy.
    Use to reduce boilerplate.

    Parameters
    ----------
    f : callable
        The objective function. Accepts an (n,d) array consisting of n points
        in d dimensional space, and returns an (n,) array of responses.
    f_jac : callable
        The objective function and it's Jacobian. Accepts a single d
        dimensional point represented by a vector of size (d,), and returns a
        tuple (y, J); y being a scalar representing the value of the objective
        function, and J being a vector of size (d,) describing it's
        derivatives with respect to each input dimension.
    d : int
        Scalar describing the dimension of the search space.
    x0 : {ndarray of shape (d,), None}, optional
        Initial point. Default=None.
    grid : {ndarray of size (n_grid, d), None}, optional
        Grid. Default=None.
    grid_size : {int, None}, optional
        Grid size. Defaults to 100*d if unspecified.
    grid_method : str, optional
        Grid method. Default='lhs'.
    grid_options : dict, optional
        Additional options to pass to the sampling algorithm described by
        'grid method'. Default=None.
    seed : {None, int32}
        Random seed. Default=None.

    Returns
    -------
    result : dict
    """
    if x0 is None:
        x0 = grid_search(
            func,
            index_dimension,
            grid,
            grid_size,
            grid_method,
            grid_options,
            seed
        )
    result = minimize(
        fun=func_and_jac,
        x0=x0,
        jac=True,
        method='SLSQP',
        bounds=[(0, 1)] * index_dimension
    )
    if result['success']:
        ret = {
            'xprime': result['x'],
            'val': result['fun'],
        }
    else:
        ret = {'raw': result}
    return ret


def grid_search(  # pylint: disable=too-many-arguments
        func: ta.ObjectiveFunction,
        index_dimension: int,
        grid: Optional[np.ndarray] = None,
        grid_size: Optional[int] = None,
        grid_method: str = 'lhs',
        grid_options: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
) -> np.ndarray:
    """ Preliminary grid search.

    Evaluate f over a (gridsize, d) grid of points generated using grid_method
    and return the point corresponding to the minimum value.

    Parameters
    ----------
    func : callable
        The objective function to minimise.
    index_dimension : int
        The dimension of the space on which the objective function is defined.
    grid : {ndarray, None}, optional
        The grid on which to perform the search. If None, a grid will be
        generated according to grid_size and grid_method. Default = None.
    grid_size : {int, None}, optional
        Grid size. If None, grid_side = d * 100
    grid_method : str, optional
        Algorithm used to generate the grid. References one of the methods
        made available via apricot.core.sampling.sample_hypercube:
        * 'lhs' Latin Hypercube Sample
        * 'olhs' Optimised Latin Hypercube Sample
        * 'mdurs' Multi-dimensionally Uniform Random Sample
        * 'sobol' Sobol sequence
        * 'randomised_sobol' Sobol sequence with randomisation
        * 'urandom' Uniform random sampling
        * 'factorial' Factorial design
        Due to grid_search performing effectively when grid_size is large,
        'olhs' and 'mdurs' are not recommended. Default = 'lhs'.
    grid_options : {dict, None}, optional
        Additional options to pass to the grid generation algorithm. Default
        = None.
    seed : {Int32, None}, optional
        Random seed.

    Returns
    -------
    x0 : ndarray
        (d,) array representing the point on the grid at which f is a minimum.

    See Also
    --------
    apricot.core.sampling.sample_hypercube
    """
    xgrid = _get_grid(
        index_dimension,
        grid=grid,
        grid_size=grid_size,
        grid_method=grid_method,
        grid_options=grid_options,
        seed=seed
    )
    fxgrid = func(xgrid)
    return xgrid[fxgrid.argmin(), :]


def _check_grid(grid: np.ndarray, index_dimension: int):
    """ Check / format a grid of points. """
    xgrid = np.atleast_1d(grid)
    if index_dimension == 1:
        xgrid = _check_grid_shape_1d(xgrid)
    else:
        xgrid = _check_grid_shape_nd(xgrid, index_dimension)
    return _check_bounds(xgrid)


def _check_bounds(xgrid: np.ndarray) -> np.ndarray:
    """Check everything is between 0 and 1.

    Parameters
    ----------
    xgrid : ndarray
        The grid of points to assess.

    Returns
    -------
    xgrid : ndarray
        The assessed grid of points.

    Raises
    ------
    ValueError
        If any of the grid points do not lie on [0,1]^d

    """
    if (xgrid > 1).any() | (xgrid < 0).any():
        raise ValueError('One or more points do not lie on [0, 1]^d')
    return xgrid


def _check_grid_shape_1d(xgrid: np.ndarray) -> np.ndarray:
    """Check the shape of a 1-dimensional grid.

    Parameters
    ----------
    xgrid : ndarray
        (n, d_grid) grid of points to assess.

    Returns
    -------
    xgrid : ndarray
        (n, 1) grid of F-ordered points.

    Raises
    ------
    ValueError
        If d_grid != 1
        If np.squeeze(xgrid).ndim > 1
    """
    if xgrid.ndim == 1:
        return xgrid.reshape(-1, 1, order='F')
    if xgrid.squeeze().ndim == 1:
        return _check_grid_shape_1d(xgrid.squeeze())
    msg = 'supplied grid should have strictly 1 non-singleton dimension.'
    raise ValueError(msg)


def _check_grid_shape_nd(
        xgrid: np.ndarray,
        index_dimension: int
) -> np.ndarray:
    """Check the shape of a d-dimensional grid.

    Parameters
    ----------
    xgrid : ndarray
        (n, d_grid) grid of points to assess.
    index_dimension : int
        The dimension the grid should be in axis 1.

    Returns
    -------
    xgrid : ndarray
        (n, index_dimension) array of F-ordered points.

    Raises
    ------
    ValueError
        If either d_grid != index_dimension or np.squeeze(xgrid) > 2.
    """
    if xgrid.ndim != 2:  # squeeze grid if it has over 2 dimensions
        grid_squeezed = xgrid.squeeze()
        if grid_squeezed.ndim == 1:
            if grid_squeezed.shape[0] == index_dimension:
                return grid_squeezed.reshape(1, -1, order='F')
            msg = 'supplied grid must be of shape (n,{0})'.format(
                index_dimension
            )
            raise ValueError(msg)
        if grid_squeezed.ndim == 2:
            return _check_grid_shape_nd(grid_squeezed, index_dimension)
        msg = 'supplied grid may have at most 2 non-singleton dimensions.'
        raise ValueError(msg)
    return _check_grid_shape_nd_internal(xgrid, index_dimension)


def _check_grid_shape_nd_internal(
        xgrid: np.ndarray,
        index_dimension: int
) -> np.ndarray:
    """ Check the shape of a 2D grid in axis 1 matches d

    Parameters
    ----------
    xgrid : ndarray
        (n, d_grid) array
    index_dimension : int
        The dimension the grid should be in axis 1.

    Returns
    -------
    xgrid : ndarray
        (n, d) array of F-ordered points.

    Raises
    ------
    ValueError
        If d_grid != d.
    """
    if xgrid.shape[1] == index_dimension:
        return force_f_array(xgrid)
    msg = 'supplied grid must be of shape (n,{0})'.format(index_dimension)
    raise ValueError(msg)


def _get_grid(  # pylint: disable=too-many-arguments
        index_dimension: int,
        grid: Optional[np.ndarray] = None,
        grid_size: Optional[int] = None,
        grid_method: str = 'lhs',
        grid_options: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
) -> np.ndarray:
    """

    Parameters
    ----------
    index_dimension: int
        Number of grid dimensions.
    grid: ndarray, optional
        Pre-generated grid of points. If absent, a grid of size
        grid_size will be generated using grid_method. Default = None.
    grid_size: int, optional
        If grid is None, generate a grid of size (grid_size, d) using
        grid_method. Default = 100 * d. Ignored if grid != None.
    grid_method: str, optional
        If grid is None, generate a grid of size (grid_size, d) using
        grid_method. Default = 'lhs'. Ignored if grid != None.
    grid_options: dict, optional
        Optional extra arguments to pass to grid_method. Default = None.
        Ignored if grid != None.
    seed: {int32, None} optional
        Random seed. Default = None.

    Returns
    -------
    grid: ndarray
        (grid_size, d) array of points.

    Raises
    ------
    ValueError
        If either the grid has more than 2 non-singleton dimensions or if the
        dimensions of grid are not consistent with the index dimension.
    """
    if grid is None:
        if grid_size is None:
            grid_size = 100 * index_dimension
        return sample_hypercube(
            grid_size,
            index_dimension,
            method=grid_method,
            seed=seed,
            options=grid_options
        )
    return _check_grid(grid, index_dimension)
