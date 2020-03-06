import numpy as np
from scipy.optimize import minimize
from apricot.core.utils import _force_f_array
from apricot.core.sampling import sample_hypercube

def optimise(f, f_jac, d, x0=None, grid=None, grid_size=None, grid_method='lhs',
             grid_options=None, seed=None):
    """ Generic numerical optimisation routine using SLSQP

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
        ret = {
            'raw' : result,
        }

    return ret

def grid_search(f, d, grid=None, grid_size=None, grid_method='lhs', grid_options=None, seed=None):
    """ Preliminary grid search"""
    xgrid = _get_grid(d, grid=None, grid_size=None, grid_method='lhs', grid_options=None, seed=seed)
    fxgrid = f(xgrid)
    return xgrid[fxgrid.argmin(),:]

def _check_grid(grid, d):
    """ Check / format a grid of points"""
    xgrid = np.atleast_1d(grid)
    if d == 1:
        xgrid = _check_grid_shape_1d(xgrid)
    else:
        xgrid = _check_grid_shape_nd(xgrid, d)
    return _check_bounds(xgrid)

def _check_bounds(xgrid):
    """Check everything is between 0 and 1"""
    if (xgrid > 1).any() | (xgrid < 0).any():
        raise ValueError('One or more points do not lie on [0, 1]^d')
    else:
        return xgrid
    
def _check_grid_shape_1d(xgrid):
    """Check the shape of a 1-dimensional grid"""
    
    # xgrid is a vector
    if xgrid.ndim == 1:
        return xgrid.reshape(-1, 1, order='F')
    
    # not a vector but should only have 1 non singleton dimension
    elif xgrid.squeeze().ndim == 1:
        return _check_grid_shape_1d(xgrid.squeeze())
    
    # incompatible
    # TODO improve this message
    else:
        raise ValueError
        
def _check_grid_shape_nd(xgrid, d):
    """Check the shape of a d-dimensional grid"""
    
    # incompatible number of dimensions
    if xgrid.ndim != 2:
        xgs = xgrid.squeeze()
        
        # grid is just a single point; weird but technically valid
        if xgs.ndim == 1:
            if xgs.shape[0] == d:
                return xgs.reshape(1, -1, order='F')
            else:
                # incompatible
                # TODO improve this message
                raise ValueError
          
        # xgrid is 2d after squeeze; check squeezed grid
        elif xgs.ndim == 2:
            return _check_grid_shape_nd(xgs, d)
        
        # incompatible; too many non-singleton dimensions
        # TODO improve this message
        else:
            raise ValueError
    
    # dimensions of grid match; check compatibility
    else:
        return _check_grid_shape_nd_internal(xgrid, d)

def _check_grid_shape_nd_internal(xgrid, d):
    if xgrid.shape[1] == d:
        return _force_f_array(xgrid)
    else:
        # TODO fix this exception (print information)
        raise ValueError

def _get_grid(d, grid=None, grid_size=None, grid_method='lhs', grid_options=None, seed=None):
    if grid is None:
        if grid_size is None:
            grid_size = 100*d
        grid = sample_hypercube(grid_size, d, method=grid_method, seed=seed, options=grid_options)
    else:
        grid = _check_grid(grid, d)
    return grid
