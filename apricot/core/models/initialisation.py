# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import Optional, Union, Dict, List, Any, Mapping, cast, Literal
import numpy as np  # type: ignore
from apricot.core.models import type_aliases as ta
from apricot.core.logger import get_logger
from apricot.core.models.prior import ls_inv_gamma

# satisfy forward type checks
if False:
    import apricot


LOGGER = get_logger()


def make_pystan_dict(
        interface_instance: 'apricot.core.models.interface.Interface',
        x: np.ndarray,
        y: np.ndarray,
        jitter: float = 1e-10,
        ls_options: ta.LsPriorOptions = None,
        seed: Optional[int] = None,
) -> ta.PyStanData:
    """ Make the 'data' dictionary to be supplied to the pystan model.

    The data dictionary contains the user-supplied information required to
    run either of pyStan's 'sampling' or 'optimizing' methods for the
    interfaced model.

    Parameters
    ----------
    interface_instance : apricot.core.models.interface.Interface
        pyStan model interface.
    x : ndarray
        (n, d) array of sample points.
    y : ndarray
        (n,) array of responses.
    jitter : float, optional
        Variance of stability jitter (square root this quantity for a standard
        deviation). Default = 1e-10.
    ls_options : {None, str, list of str}, optional
        Additional fit options for each dimension of the index. If fit_options
        is a list, it must be of length d, where d is the dimension of the
        model's input space. If fit_options is a string, the requested options
        will be applied to all of the model's input dimensions. Default = None.
    seed : {None, int}
        Seed for numpy's random state. If not provided, an arbitrary seed will
        be used. Default = None.

    Returns
    -------
    pyStan_data_dict : dict
        The 'data' dictionary such that interface_instance.pystan_model can
        invoke either its sampling or optimizing methods with data=data.
    """

    n, d = x.shape
    data: Dict[Any, Union[int, float, np.ndarray]] = {
        'x': x,
        'y': y,
        'n': n,
        'd': d,
        'jitter': jitter,
    }

    # eq_flat does not have hyperparameter priors
    if interface_instance.kernel_type != 'eq_flat':
        # lengthscales use inverse gamma hyperprior
        ls_alpha, ls_beta = ls_inv_gamma.ls_inv_gamma_prior(
            x,
            ls_options,
            seed=seed
        )
        data['ls_alpha'] = ls_alpha
        data['ls_beta'] = ls_beta

        # amplitudes have a t (3, 0, 1)
        data['amp_loc'] = 0.0
        data['amp_scale'] = 1.0

    # rational quadratic has an additional hyperparameter kappa
    if interface_instance.kernel_type == 'rq':
        data['kappa_loc'] = 0.0
        data['kappa_scale'] = 1.0

    # set up prior for beta if using a linear mean
    if interface_instance.mean_function_type == 'linear':
        data['beta_loc'] = np.zeros(d+1, dtype=np.float64)
        data['beta_scale'] = np.ones(d+1, dtype=np.float64)

    # set xi to Normal(0, xi_scale) if inferring xi, where xi_scale = sd(y)/10
    if interface_instance.noise_type[0] == 'infer':
        data['xi_scale'] = np.std(y) / 10.0

    elif interface_instance.noise_type[0] == 'deterministic':
        data['xi'] = interface_instance.noise_type[1]

    if interface_instance.warping:
        if interface_instance.warping == 'linear':
            data['alpha_warp_mu'] = np.zeros(d, dtype=np.float64)
            data['alpha_warp_sigma'] = np.full(d, 0.5)
            data['beta_warp_mu'] = np.zeros(d, dtype=np.float64)
            data['beta_warp_sigma'] = np.full(d, 0.5)
        elif interface_instance.warping == 'sigmoid':
            data['alpha_warp_mu'] = np.full(d, 2.0)
            data['alpha_warp_sigma'] = np.full(d, 0.5)
            data['beta_warp_mu'] = np.full(d, 2.0)
            data['beta_warp_sigma'] = np.full(d, 0.5)

    cast(ta.PyStanData, data)
    return data


def get_init(
        interface_instance: 'apricot.core.models.interface.Interface',
        init: Optional[ta.InitTypes],
        stan_dict: ta.PyStanData,
) -> ta.InitTypes:
    """ Invoke various initialisation methods for the sampler.

    Parameters
    ----------
    interface_instance : apricot.core.models.interface.Interface
        pyStan model interface.
    stan_dict : dict
        The "data" dictionary to be passed to pyStan.
    init : {None, 'stable', 'random', dict}
        Determines initialisation method:
        * None : defaults to 'stable'.
        * 'stable' : "stable" initialise parameters from data:
            - The marginal standard deviation is initialised to the sample
                marginal stabard deviation of the supplied function responses,
                that is amp_init = std(y).
            - Lengthscales are initialised to the standard deviation of
                the sample points in their respective dimensions, that is
                ls_init_i = std(X, axis=0)[i]
            - The noise standard deviation (if present) is initialised to
                 1/10th of the marginal standard deviation.
            - Warping parameter are initialised to values corresponding to the
                 desired type of warping.
        * 'zero' : initialise all parameters from zero.
        * 'random' : initialise all parameters randomly on their support.
        * dict : A custom initialisation value for each of the model's
            parameters.

    Returns
    -------
    init : {dict, str, int}
        Initialisation values for the parameters for the desired
        pyStan model. Either a dictionary, 'random' or 0.

    See Also
    --------
    initialisation._init_from_data
    """
    if init is None:
        init = 'stable'

    # custom init, let pyStan raise its own exceptions if necessary
    if isinstance(init, dict):
        LOGGER.debug('Initialisation: user.')
        return init
    elif isinstance(init, str):
        if init.lower() == 'stable':
            LOGGER.debug('Initialisation: stable.')
            return init_from_data(interface_instance, stan_dict)
        else:
            LOGGER.debug('Initialisation: {0}'.format(init))
            return init_from_str(init)
    elif isinstance(init, list):
        # TODO: handle case in which init is a List. Must match
        # the number of requested chains!
        pass
    raise TypeError(
        'Unable to parse init option of type "{0}".'.format(type(init))
    )


def init_from_data(
        interface_instance: 'apricot.core.models.interface.Interface',
        stan_dict: ta.PyStanData
) -> ta.InitData:
    """ Initialise from data.

    Use the data to identify suitable initialisation values for HMC.

    Parameters
    ----------
    interface_instance : instance of apricot.core.models.interface.Interface
        pyStan model interface
    stan_dict : dict
        The pyStan 'data' dict.

    Returns
    -------
    init : dict
        Dictionary of initialisation values for the sampler.
    """
    x = stan_dict['x']
    y = stan_dict['y']
    d = stan_dict['d']
    init: Dict[Any, Union[np.ndarray, float]] = {'ls': np.std(x, axis=0) / 3.0}
    init['amp'] = np.std(y)
    if interface_instance.noise_type[0] == 'infer':
        init['xi'] = np.std(y) / 10.0
    if interface_instance.mean_function_type == 'linear':
        init['beta'] = np.zeros(d+1, dtype=np.float64)
    if interface_instance.warping:
        if interface_instance.warping == 'sigmoid':
            init['alpha_warp'] = np.full(d, 5.0)
            init['beta_warp'] = np.full(d, 5.0)
        if interface_instance.warping == 'linear':
            init['alpha_warp'] = np.ones(d, dtype=np.float64)
            init['beta_warp'] = np.ones(d, dtype=np.float64)
    # to satisfy type checker
    cast(ta.InitData, init)
    return init


def init_from_str(init_str: str) -> Union[str, int]:
    """

    Parameters
    ----------
    init_str : str
        Requested (Stan compatible) initialisation method.

    Returns
    -------
    init_method : str, int
        Either 'random' or 0.

    Raises
    ------
    ValueError
       If init_str is not in {'random', '0', 'zero'}.
    """
    options: Mapping[str, Union[str, int]] = {
        'random': 'random',
        '0': 0,
        'zero': 0,
    }
    try:
        return options[init_str]
    except KeyError:
        msg = 'Could not find init option matching "{0}".'.format(init_str)
        raise ValueError(msg) from None
