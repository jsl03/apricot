import typing
import numpy as np

from apricot.core import utils

def run_mle(
        interface : 'apricot.core.models.interface.Interface',
        x : np.ndarray,
        y : np.ndarray,
        jitter : float = 1e-10,
        fit_options : typing.Optional[dict] = None,
        init_method : str = 'random',
        algorithm : str = 'Newton',
        restarts : int = 10,
        max_iter : int = 250,
        seed : typing.Optional[int] = None,
):
    """ Maximum marginal likelihood estimation via numerical optimisation

    Parameters
    ----------
    interface : instance of models.interface.Interface
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
    init_method : {'stable', 'zero', 'random'}
        String determining the initialisation method:
        * 'stable' : "stable" initialise from data.
        * 'zero' : initialise all parameters from zero.
        * 'random' : initialise all parameters randomly on their support.
        Separate random initialisations are used for each restart.
        Default = 'random'.
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
        Seed for numpy's random state. Also used to initialise pyStan.
        Default = None.

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
        seed = utils.random_seed()

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
        'theta_init' : init,
        'max_iter' : max_iter,
        'restarts' : restarts,
        'lp': result['value']
    }
    return parameters, info

def _mle_internal(
        interface : 'apricot.core.models.interface.Interface',
        opts : dict,
        restarts : int,
):
    """ Interface to Stan's optimiser. """
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
            raise RuntimeError(rte) from None
    return result
