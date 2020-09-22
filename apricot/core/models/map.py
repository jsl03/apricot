"""
Code for fitting a model using Maximum A-Priori Probability
(a.k.a. type II maximum likelihood estimation)

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""

from typing import cast, Optional, Any, Dict, Mapping, Tuple
import numpy as np  # type: ignore
from apricot.core import utils
from apricot.core.logger import get_logger
from apricot.core.models import type_aliases as ta


# for satisfying forward type checking
if False:  # pylint: disable=using-constant-test
    import apricot  # pylint: disable=unused-import


LOGGER = get_logger()


def run_map(  # pylint: disable=too-many-arguments, too-many-locals
        interface: 'apricot.core.models.interface.Interface',
        x_data: np.ndarray,
        y_data: np.ndarray,
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
    x_data : ndarray
        (n,d) array with each row representing a sample point in d-dimensional
        space.
    y_data : ndarray
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

    Raises
    ------
    RuntimeError
        If the optimisation routine failed.
    """

    if seed is None:
        seed = utils.random_seed()
    data = interface.make_pystan_dict(x_data, y_data, jitter, ls_options)
    init = interface.get_init(init_method, data)
    opts: Mapping[str, Any] = {
        'data': data,
        'init': init,
        'as_vector': False,
        'algorithm': algorithm,
        'iter': max_iter,
        'seed': seed,
    }
    result = map_internal(interface, opts, restarts)
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
        return parameters, info
    msg = 'Failed to identify a solution.'
    raise RuntimeError(msg)


def _assign_new_seed(opts: Dict[str, Any]) -> Dict[str, Any]:
    current_seed = opts['seed']
    new_seed = utils.random_seed(current_seed)
    opts['seed'] = new_seed
    return opts


def map_internal(
        interface: 'apricot.core.models.interface.Interface',
        opts: Mapping[str, Any],
        restarts: int,
) -> Dict[str, Any]:
    """ Interface to Stan's optimiser. """
    best = -np.inf
    result = None
    exceptions = []
    # prevent mutating original options dictionary in case we need it
    opts_n = cast(Dict[str, Any], opts).copy()
    for rep in range(restarts):
        if rep > 0:
            opts_n = _assign_new_seed(opts_n)
            LOGGER.debug(
                'Restarting optimiser (r = %(rep)s) new seed: %(seed)s',
                {'rep': rep, 'seed': opts_n['seed']}
            )
        try:
            result = interface.pystan_model.optimizing(**opts_n)
            LOGGER.debug(
                'Optimiser exited normally [lp__ : %(value)s]',
                {'value': result['value']}
            )
            if result['value'] > best:
                best = result['value']
                # result should only be reassigned if value > best
                result = result  # pylint: disable=self-assigning-variable
        except RuntimeError as rte:
            LOGGER.debug(
                'RuntimeError encountered (restart # %(rep)s): %(rte)s',
                {'rep': rep, 'rte': rte}
            )
            exceptions.append(rte)
        if (rep == 0) & (restarts > 1):
            opts_n['init'] = 'random'
            LOGGER.debug(
                'Subsequent initialisations changed to random (repeats > 1)'
            )
    if result is None:
        errs = '\n' + '\n'.join([str(exception) for exception in exceptions])
        msg = 'Converence could be achieved: {0}'.format(errs)
        raise RuntimeError(msg)
    return result
