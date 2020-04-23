# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
import typing
import numpy as np
from apricot.core import utils
from apricot.core.logger import get_logger, log_options


logger = get_logger()


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
    logger.debug(
        'Initialising optimisation routine with: \n{0}'.format(
            log_options(opts)
        )
    )
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


def _assign_new_seed(opts):
    current_seed = opts['seed']
    new_seed = utils.random_seed(current_seed)
    opts['seed'] = new_seed
    return opts


def _mle_internal(
        interface: 'apricot.core.models.interface.Interface',
        opts: dict,
        restarts: int,
) -> dict:
    """ Interface to Stan's optimiser. """
    best = -np.inf
    result = None

    # prevent mutating original options dictionary in case we need it
    # somewhere later (where?)
    opts_n = opts.copy()

    for r in range(restarts):
        # prevent using the same seed repeatedly, but make sure logger can keep
        # track of state
        if r > 0:
            opts_n = _assign_new_seed(opts_n)
            logger.debug('Restarting optimiser (r = {0}) new seed: {1}'.format(
                r, opts_n['seed']
            ))

        # sometimes the optimiser exits with runtime error, due to eg.
        # line search failure (l-bfgs only) or cholesky decompose being
        # called on something that isnt PSD
        try:
            result = interface.pystan_model.optimizing(**opts_n)
            logger.debug(
                'Optimiser exited normally [lp__ : {0}]'.format(
                    result['value']
                )
            )
            if result['value'] > best:
                best = result['value']
                result = result

        except RuntimeError as rte:
            logger.debug(
                'RuntimeError encountered (r = {1}): {2}'.format(
                    opts_n['algorithm'], r, rte
                )
            )
            exception = rte

        if (r == 0) & (restarts > 1):
            opts_n['init'] = 'random'
            logger.debug(
                'Subsequent initialisations changed to random (repeats > 1)'
            )
           
    # TODO at the moment this stores only the *last* exception
    # encountered above. Better to raise custom RTE and print a list?
    if result is None:
        raise exception

    return result
