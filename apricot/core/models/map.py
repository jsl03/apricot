# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import cast, Union, Optional, Any, Dict, Mapping, Tuple
import numpy as np  # type: ignore
from apricot.core import utils
from apricot.core.logger import get_logger
from apricot.core.models import type_aliases as ta


# satisfy forward type checks
if False:
    import apricot


LOGGER = get_logger()


def run_map(
        interface: 'apricot.core.models.interface.Interface',
        x: np.ndarray,
        y: np.ndarray,
        jitter: float = 1e-10,
        ls_options: Optional[ta.LsPriorOptions] = None,
        init_method: ta.InitTypes = 'random',
        algorithm: str = 'Newton',
        restarts: int = 10,
        max_iter: int = 250,
        seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ Maximum a-posteriori probability hyperparameter identification via
    numerical optimisation.

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
    ls_options :
        Optional extra parameters to the GP prior distribution.
    init_method : {'stable', 'zero', 'random', None}
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
    Interface.map
    """

    if seed is None:
        seed = utils.random_seed()
    data = interface.make_pystan_dict(x, y, jitter, ls_options)
    init = interface.get_init(init_method, data)
    opts: Mapping[str, Any] = {
        'data': data,
        'init': init,
        'as_vector': False,
        'algorithm': algorithm,
        'iter': max_iter,
        'seed': seed,
    }
    result = _map_internal(interface, opts, restarts)
    if result is not None:
        parameters = result['par']
        info = {
            'method': 'map',
            'algorithm': algorithm,
            'theta_init': init,
            'max_iter': max_iter,
            'restarts': restarts,
            'lp': result['value']
        }
    else:
        # TODO
        parameters = None
        info = None
    return parameters, info


def _assign_new_seed(opts: Dict[str, Any]) -> Dict[str, Any]:
    current_seed = opts['seed']
    new_seed = utils.random_seed(current_seed)
    opts['seed'] = new_seed
    return opts


def _map_internal(
        interface: 'apricot.core.models.interface.Interface',
        opts: Mapping[str, Any],
        restarts: int,
) -> Dict[str, Any]:
    """ Interface to Stan's optimiser. """
    best = -np.inf
    result = None
    # prevent mutating original options dictionary in case we need it 
    opts_n = cast(Dict[str, Any], opts).copy()
    for r in range(restarts):
        if r > 0:
            opts_n = _assign_new_seed(opts_n)
            LOGGER.debug('Restarting optimiser (r = {0}) new seed: {1}'.format(
                r, opts_n['seed']
            ))
        try:
            result = interface.pystan_model.optimizing(**opts_n)
            LOGGER.debug(
                'Optimiser exited normally [lp__ : {0}]'.format(
                    result['value']
                )
            )
            if result['value'] > best:
                best = result['value']
                result = result
        except RuntimeError as rte:
            LOGGER.debug(
                'RuntimeError encountered (r = {0}): {1}'.format(
                    r, rte
                )
            )
            exception = rte
        if (r == 0) & (restarts > 1):
            opts_n['init'] = 'random'
            LOGGER.debug(
                'Subsequent initialisations changed to random (repeats > 1)'
            )
    # TODO at the moment this stores only the *last* exception
    # encountered above. Better to raise custom RTE and print a list?
    if result is None:
        raise exception
    return result
