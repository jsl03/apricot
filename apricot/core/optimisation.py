import typing
import numpy as np
from scipy.optimize import minimize
from apricot.core.utils import _force_f_array
from apricot.core.sampling import sample_hypercube

def optimise(
        f : callable, 
        f_jac : callable,
        d : int,
        x0 : typing.Optional[np.ndarray] = None,
        grid : typing.Optional[np.ndarray] = None,
        grid_size : typing.Optional[int] = None,
        grid_method : str = 'lhs',
        grid_options : typing.Optional[dict] = None,
        seed : typing.Optional[int] = None,
):
    """ Generic numerical optimisation routine using SLSQP.

    Generic numerical optimisation strategy consisting of a preliminary grid
    search and a gradient based optimisation algorithm implemented in scipy.
    Use to reduce boilerplate.

    Parameters
    ----------
    f : callable
        The objective function. Accepts an (n,d) array consisting of n points in
        d dimensional space, and returns an (n,) array of responses.
    f_jac : callable
        The objective function and it's Jacobian. Accepts a single d dimensional
        point represented by a vector of size (d,), and returns a tuple (y, J);
        y being a scalar representing the value of the objective function, and
        J being a vector of size (d,) describing it's derivatives with respect
        to each input dimension
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

    # if no initial point is provided, run a grid search
    if x0 is None:
        opts = {
            'grid' : grid,
            'grid_size' : grid_size,
            'grid_method' : grid_method,
            'grid_options' : grid_options,
            'seed' : seed,
        }
        x0 = grid_search(f, d, **opts)

    # run the optimiser
    result = minimize(fun=f_jac, x0=x0, jac=True, method='SLSQP', bounds=[(0,1) for _ in range(d)])

    # format results
    if result['success']:
        ret = {
            'xprime' : result['x'],
            'val' : result['fun'],
        }
    else:
        ret = {'raw' : result}

    return ret

def grid_search(
        f : callable,
        d : int,
        grid : typing.Optional[np.ndarray] = None,
        grid_size : typing.Optional[int] = None,
        grid_method : str ='lhs',
        grid_options : typing.Optional[dict] = None,
        seed : typing.Optional[int] = None,
):
    """ Preliminary grid search.

    Evaluate f over a (gridsize, d) grid of points generated using grid_method
    and return the point corresponding to the minimum value.

    Parameters
    ----------
    f : callable
        The objective function to minimise.
    d : int
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
    xgrid = _get_grid(d, grid=None, grid_size=None, grid_method='lhs', grid_options=None, seed=seed)
    fxgrid = f(xgrid)
    return xgrid[fxgrid.argmin(),:]

def _check_grid(grid : np.ndarray, d : int):
    """ Check / format a grid of points. """
    xgrid = np.atleast_1d(grid)
    if d == 1:
        xgrid = _check_grid_shape_1d(xgrid)
    else:
        xgrid = _check_grid_shape_nd(xgrid, d)
    return _check_bounds(xgrid)

def _check_bounds(xgrid : np.ndarray):
    """Check everything is between 0 and 1. """
    if (xgrid > 1).any() | (xgrid < 0).any():
        raise ValueError('One or more points do not lie on [0, 1]^d')
    else:
        return xgrid
    
def _check_grid_shape_1d(xgrid : np.ndarray):
    """Check the shape of a 1-dimensional grid. """
    
    if xgrid.ndim == 1:
        return xgrid.reshape(-1, 1, order='F')
    
    elif xgrid.squeeze().ndim == 1:
        return _check_grid_shape_1d(xgrid.squeeze())
    
    # TODO fix 
    else:
        raise ValueError
        
def _check_grid_shape_nd(xgrid : np.ndarray, d : int):
    """Check the shape of a d-dimensional grid. """
    if xgrid.ndim != 2:  # squeeze grid if it has over 2 dimensions
        xgs = xgrid.squeeze()
        if xgs.ndim == 1:
            if xgs.shape[0] == d:
                return xgs.reshape(1, -1, order='F')
            else:  # incompatible
                # TODO improve this message
                raise ValueError 
        elif xgs.ndim == 2:  # xgrid is 2d after squeeze; check squeezed grid
            return _check_grid_shape_nd(xgs, d)
        else:  # incompatible; too many non-singleton dimensions
            raise ValueError
    else:  # dimensions of grid match; check compatibility
        return _check_grid_shape_nd_internal(xgrid, d)

def _check_grid_shape_nd_internal(xgrid : np.ndarray, d : int):
    if xgrid.shape[1] == d:
        return _force_f_array(xgrid)
    else:
        # TODO fix this exception (print information)
        raise ValueError

def _get_grid(
        d : int,
        grid : typing.Optional[np.ndarray] = None,
        grid_size : typing.Optional[int] = None,
        grid_method : str = 'lhs',
        grid_options : typing.Optional[dict] = None,
        seed : typing.Optional[int] = None,
):
    if grid is None:
        if grid_size is None:
            grid_size = 100*d
        grid = sample_hypercube(grid_size, d, method=grid_method, seed=seed, options=grid_options)
    else:
        grid = _check_grid(grid, d)
    return grid
