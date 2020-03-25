import typing

import numpy as np
from apricot.core.models.prior import ls_inv_gamma

Fit_Options_Type_Alias = typing.Optional[typing.Union[str, typing.List[str]]]


# TODO: refactor
def make_pystan_dict(
        interface_instance: 'apricot.core.models.interface.Interface',
        x: np.ndarray,
        y: np.ndarray,
        jitter: float = 1e-10,
        fit_options: Fit_Options_Type_Alias = None,
        seed: typing.Optional[int] = None,
):
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
    fit_options : {None, str, list of str}, optional
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
    data = {
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
            fit_options,
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

    # otherwise xi is fixed to a supplied value
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

    return data


def get_init(
        interface_instance: 'apricot.core.models.interface.Interface',
        stan_dict: dict,
        init: typing.Optional[typing.Union[dict, typing.List[str], str]],
):
    """ Invoke various initialisation methods for the sampler.

    Parameters
    ----------
    interface_instance : apricot.core.models.interface.Interface
        pyStan model interface.
    stan_dict : dict
        The "data" dictionary to be passed to pyStan.
    init : {None, dict, list of str, str}
        * if init is a dict, it is assumed to contain an initial value for each
        of the parameters to be sampled by the pyStan model.
        * if init is None, the init method defaults to 'stable'.
        * if init is a string, it is matched to one of the following:
            - 'stable' : initialise from data. Lengthscales are initialised to
            the standard deviation of the respective column of x.
            - 'zero' : initialise all parameters as 0.
            - 'random' : initialise all parameters randomly on their
            support.
        * if init is a list, it is assumed to contain a string (see above) for
        each of the sample chains.

    Returns
    -------
    init : {dict, str}
        Initialisation values for the parameters for the desired
        pyStan model.

    See Also
    --------
    initialisation._init_from_data
    """
    if init is None:
        init = 'stable'
    # custom init, let pyStan raise it's own exceptions if necessary
    if type(init) is dict:
        return init
    # match init to a valid option or throw an exception
    elif type(init) is str:
        # initialising from data requires a function
        if init.lower() == 'stable':
            return _init_from_data(interface_instance, stan_dict)
        # otherwise match the string to a valid option
        else:
            return _init_from_str(init)
    # if we fall through to here, init is not a string or a dictionary
    raise TypeError(
        'Unable to parse init option of type "{0}".'.format(type(init))
    )


#TODO refactor
def _init_from_data(
        interface_instance: 'apricot.core.models.interface.Interface',
        stan_dict: dict
):
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
    init = {'ls': np.std(x, axis=0)}
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
    return init


def _init_from_str(init_str: str):
    options = {
        'random': 'random',
        '0': 0,
        'zero': 0,
    }
    try:
        return options[init_str]
    except KeyError:
        raise ValueError('Could not find init option matching "{0}".'
                         .format(init_str)) from None
