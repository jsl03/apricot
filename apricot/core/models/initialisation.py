import numpy as np
from apricot.core.models.prior.ls_inv_gamma import ls_inv_gamma_prior

def make_pystan_dict(model, x, y, jitter=1e-10, fit_options=None, seed=None):
    """ Make the 'data' dictionary for supplying to the pystan model """

    n, d = x.shape
    data = {
        'x':x,
        'y':y,
        'n':n,
        'd':d,
        'jitter':jitter,
    }

    # expq_flat does not take hyperparameters
    if model.kernel_type != 'expq_flat':

        # lengthscales use inverse gamma hyperprior
        ls_alpha, ls_beta = ls_inv_gamma_prior(x, fit_options, seed=seed)
        data['ls_alpha'] = ls_alpha
        data['ls_beta'] = ls_beta

        # amplitudes have a t (3, 0, 1)
        data['amp_loc'] = 0.0
        data['amp_scale'] = 1.0

    # rational quadratic has an additional hyperparameter kappa
    if model.kernel_type == 'rq':
        data['kappa_loc'] = 0.0
        data['kappa_scale'] = 1.0

    # set up prior for beta if using a linear mean
    if model.mean_function_type == 'linear':
        data['beta_loc'] = np.zeros(d+1, dtype=np.float64)
        data['beta_scale'] = np.ones(d+1, dtype=np.float64)

    # set xi to Normal(0, xi_scale) if inferring xi
    if model.noise_type[0] == 'infer':
        data['xi_scale'] = np.std(y) / 10.0

    # otherwise xi is fixed to a supplied value
    elif model.noise_type[0] == 'deterministic':
        data['xi'] = model.noise_type[1]

    if model.warping:
        if model.warping == 'linear':
            data['alpha_warp_mu'] = np.zeros(d, dtype=np.float64)
            data['alpha_warp_sigma'] = np.full(d, 0.5)
            data['beta_warp_mu'] = np.zeros(d, dtype=np.float64)
            data['beta_warp_sigma'] = np.full(d, 0.5)

        elif model.warping == 'sigmoid':
            data['alpha_warp_mu'] = np.full(d, 2.0)
            data['alpha_warp_sigma'] = np.full(d, 0.5)
            data['beta_warp_mu'] = np.full(d, 2.0)
            data['beta_warp_sigma'] = np.full(d, 0.5)

    return data

def get_init(model, stan_dict, init):
    """Invoke various initialisation methods for the sampler.

    Parameters
    ----------
    stan_dict : dict
        The "data" dictionary to be passed to pystan.

    init : {dict, str}
        * if init is a dict, it is assumed to contain an initial value for each
        of the parameters to be sampled by the pyStan model.
        * if init is None, the init method defaults to 'data'.
        * if init is a string, it is matched to one of the following:
            - 'stable' : initialise from data. Lengthscales are initialised to
            the standard deviation of the respective column of x.
            - {'0' , 'zero} : initialise all parameters as 0.
            - 'random' : initialise all parameters randomly on their
            support.
    Returns
    -------
    init : {dict, str}
        Initialisation values for the parameters for the desired
        pyStan model.

    Notes
    -----
    To do: custom initialisation via a dict.
    """
    # init from data
    if init is None:
        init = 'stable'

    # custom init, let pyStan raise it's own exceptions if necessary
    if type(init) is dict:
        return init

    # match init to a valid option or throw an exception
    elif type(init) is str:

        # initialising from data requires a function
        if init.lower() == 'stable':
            return _init_from_data(model, stan_dict)

        # otherwise match the string to a valid option
        else:
            return _init_from_str(init)

    # if we fall through to here, init is not a string or a dictionary
    raise TypeError('Unable to parse init option of type "{0}".'.format(type(init)))

def _init_from_data(model, stan_dict):

    # get some variables from the StanDict
    x = stan_dict['x']
    y = stan_dict['y']
    d = stan_dict['d']

    init = {'ls' : np.std(x, axis=0)}
    init['amp'] = np.std(y)

    if model.noise_type[0] == 'infer':
        init['xi'] = np.std(y) / 10.0

    if model.mean_function_type == 'linear':
        init['beta'] = np.zeros(d+1, dtype=np.float64)

    if model.warping:
        if model.warping == 'sigmoid':
            init['alpha_warp'] = np.full(d, 5.0)
            init['beta_warp'] = np.full(d, 5.0)
        if model.warping == 'linear':
            init['alpha_warp'] = np.ones(d, dtype=np.float64)
            init['beta_warp'] = np.ones(d, dtype=np.float64)
    return init

def _init_from_str(init):
    options = {
        'random' : 'random',
        '0' : 0,
        'zero' : 0,
    }
    try:
        return options[init]
    except KeyError:
        raise ValueError('Could not find init option matching "{0}".'.format(init)) from None
