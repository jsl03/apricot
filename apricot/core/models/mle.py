# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
import typing
import numpy as np
from apricot.core import utils


def run_mle(
        interface: 'apricot.core.models.interface.Interface',
        x: np.ndarray,
        y: np.ndarray,
        jitter: float = 1e-10,
        fit_options:  typing.Optional[dict] = None,
        init_method: typing.Union[dict, str] = 'random',
        algorithm: str = 'Newton',
        restarts: int = 10,
        max_iter: int = 250,
        seed: typing.Optional[int] = None,
) -> typing.Union[np.ndarray, dict]:
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
        String determining the initialisation method. Note that if restarts > 1
        , only the first optimisation is initialised according to init_method,
        and the rest will be initialised using init_method = 'random':
        * 'stable' : "stable" initialise parameters from "stable" guesses.
        * 'zero' : initialise all parameters from zero.
        * 'random' : initialise all parameters randomly on their support.
        * dict : A custom initialisation value for each of the model's
            parameters.
        Default = 'random'.
    algorithm : str, optional
        String specifying which of Stan's gradient based optimisation
        algorithms to use. Default = 'Newton'.
    restarts : int, optional
        The number of restarts to use. The optimisation will be repeated
        this many times and the hyperparameters with the highest
        log-likelihood will be returned. restarts > 1 is not compatible with
        initialisation = 0. Default=10.
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
    data = interface.make_pystan_dict(x, y, jitter, fit_options)
    init = interface.get_init(init_method, data)
    opts = {
        'data': data,
        'init': init,
        'as_vector': False,
        'algorithm': algorithm,
        'iter': max_iter,
        'seed': seed,
    }
    result = _mle_internal(interface, opts, restarts)
    parameters = result['par']
    info = {
        'method': 'mle',
        'algorithm': algorithm,
        'theta_init': init,
        'max_iter': max_iter,
        'restarts': restarts,
        'lp': result['value']
    }
    return parameters, info


def _mle_internal(
        interface: 'apricot.core.models.interface.Interface',
        opts: dict,
        restarts: int,
) -> dict:
    """ Interface to Stan's optimiser. """
    best = -np.inf
    result = None

    # options for first restart
    opts_0 = opts

    # subsequent restarts always use random initialisation
    if restarts > 1:
        opts_n = opts.copy()
        opts_n['init'] = 'random'

    for r in range(restarts):

        if r == 0:
            opts_to_use = opts_0
        else:
            opts_to_use = opts_n

        try:
            result = interface.pystan_model.optimizing(**opts_to_use)
            if result['value'] > best:
                best = result['value']
                result = result

        # TODO save RTE and pass through to info dictionary
        except RuntimeError:
            pass

        # TODO fix: something prevented optimiser from succeeding
        if result is None:
            raise RuntimeError from None

    return result
