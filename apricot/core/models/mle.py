import numpy as np
from apricot.core.utils import random_seed

def run_mle(interface, x, y, jitter=1e-10, fit_options=None, init_method='random',
            algorithm='Newton', restarts=10, max_iter=250, seed=None):
    """ Maximum marginal likelihood estimation via numerical optimisation

    Parameters
    ----------
    interface : apricot.Interface instance
        Interface to the desired Stan model
    x : ndarray
        (n,d) array with each row representing a sample point in d-dimensional
        space.
    y : ndarray
        (n,) array of responses corresponding to each row of x.
    jitter : float, optional
        Magnitude of stability jitter. Default = 1e-10.
    fit_options :
        Optional extra parameters to the GP prior distribution.
    init_method : {'random', 0, 'random'}, optional
        Initialisation method for the optimisation:
        * 'random' - randomly initialise hyperparameters on their support
        * 0 - initialise all hyperparameters to 0
        * 'stable' - initialise amplitude parameters to 1, lengthscales to
        the standard deviation of the x in each respective dimension, and
        other parameters to the mean of their prior distribution. Default
        = 'random'.
    algorithm : str, optional
        String specifying which of Stan's gradient based optimisation
        algorithms to use. Default = 'Newton'.
    restarts : int, optional
        The number of restarts to use. The optimisation will be repeated
        this many times and the hyperparameters with the highest
        log-likelihood will be returned. restarts > 1 is not compatible with
        'stable' or 0 initialisations. Default=10.
    max_iter : int, optional
        Maximum allowable number of iterations for the chosen optimisation
        algorithm. Default = 250.
    seed : {int32, None}
        Random seed.

    Returns
    -------
    Parameters : dict
        Dictionary containing the optimised model hyperparameters.
    info : dict
        Diagnostic information.

    Notes
    -----
    Can be invoked as an instance method for a given interface: see
    Interface.mle
    """

    if seed is None:
        seed = random_seed()

    # make the data dictionary
    data = interface.make_pystan_dict(x, y, jitter, fit_options)

    # get the appropriate inits
    init = interface.get_init(init_method, data)

    # assign options
    opts = {
        'data':data,
        'init':init,
        'as_vector' : False,
        'algorithm': algorithm,
        'iter': max_iter,
        'seed' : seed,
    }
    result = _mle_internal(interface, opts, restarts)
    parameters = result['par']
    info = {
        'method' : 'mle',
        'algorithm' : algorithm,
        'max_iter' : max_iter,
        'restarts' : restarts,
        'lp': result['value']
    }
    return parameters, info

def _mle_internal(interface, opts, restarts):
    """Interface to pyStan optimiser"""
    best = -np.inf
    result = None
    for r in range(restarts):
        try:
            result = interface.pystan_model.optimizing(**opts)
            if result['value'] > best:
                best = result['value']
                result = result
        except RuntimeError as rte:
            # TODO save these to pass through info dictionary
            pass
        if result is None:
            raise RuntimeError(rte)
    return result
