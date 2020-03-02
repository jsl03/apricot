from apricot.core.emulator import Emulator
from apricot.core.models import Interface, glue

def fit(x, y, kernel='expq', mean=None, noise=None, method='hmc', **kwargs):
    """ Fit a Gaussian Process emulator to data.

    Given inputs, x, and outputs, y, infer the hyperparameters of a GP
    emulator with kernel 'kernel', mean function 'mean', and noise type 'noise'.

    Valid fit methods are currently 'hmc' or 'mle'.

    Parameters
    ----------
    x : ndarray
        (n,d) array with each row representing a sample point in d-dimensional
        space.
    y : ndarray
        (n,) array of responses corresponding to each row of x.
    kernel : {'expq'}
        String describing the desired kernel type:
        * 'expq' Exponentiated quadratic ("squared exponential").
    mean : {None}
        String describing the desired mean function.
        * {None, 'zero', 0} Zero mean function
    noise : {None}
        String describing the desired noise model.
        * {None, 'zero', 0} Zero (noiseless) model.
    method : {'hmc', 'mle'}
        string describing the desired fit method:
        * 'hmc' Hamiltonian Monte-Carlo. Uses Stan's implementation of the
            No-U-Turn Sampler (NUTS).
        * 'mle' Maximum Log Likelihood Estimation. Strictly speaking this aims
            to find the maximum log *marginal* likelihood using numerical
            optimisation.
    samples : int, optional
        Number of samples to draw from the posterior (accounting for the number
        of chains, warmup and thinning). Valid only if method = 'hmc'.
        Default = 2000.
    thin : int
        If > 1, keep only every thin samples. Valid only if method = 'hmc'.
        Default = 1.
    chains : int
        The number of independent chains to draw samples from. Valid only if
        method = 'hmc'. Default = 4.
    adapt_delta : float < 1
        adapt_delta control parameter for the sampler. Valid only if method =
        'hmc'. Default = 0.8.
    max_treedepth : int
        Maximum sample tree depth control parameter. Valid only if method =
        'hmc'. Default = 10.
    seed : {int32, None}
        Random seed.
    permute : bool, optional
        If True, permute the samples. Valid only if method = 'hmc'.
        Default = True.
    init_method : {'stable', 0, 'random'}
        String determining the initialisation method for each chain. Default =
        'stable'.

    Returns
    -------
    Model : apricot.Emulator instance

    """

    interface = Interface(kernel, mean, noise)

    if method.lower() == 'hmc':
        return _fit_hmc(interface, x, y, **kwargs)
    elif method.lower() == 'mle':
        return _fit_mle(interface, x, y, **kwargs)
    else:
        raise ValueError("Unrecognised fit method: '{0}'.".format(method))

def _fit_hmc(interface, x, y, jitter=1e-10, fit_options=None, samples=4000,
             thin=1, chains=4, adapt_delta=0.8, max_treedepth=10, seed=None,
             permute=True, init_method='stable'):
    """ Run Stan's HMC algorithm for the provided model.

    Parameters
    ----------
    interface : apricot.Interface instance
        Interface to the desired Stan model.
    x : ndarray
        (n,d) array with each row representing a sample point in d-dimensional
        space.
    y : ndarray
        (n,) array of responses corresponding to each row of x.
    jitter : float, optional
        Stability jitter. Default = 1e-10.
    fit_options : string or list of string, optional
        Specify either a list of 'linear' or 'nonlinear' for each input dimension,
        or a single string to apply the supplied option to all input dimensions.
    samples : int, optional
        Number of samples to draw from the posterior (accounting for the number
        of chains, warmup and thinning). Default = 2000.
    thin : int
        If > 1, keep only every thin samples. Default = 1.
    chains : int
        The number of independent chains to draw samples from. Default = 4.
    adapt_delta : float < 1
        adapt_delta control parameter for the sampler. Default = 0.8.
    max_treedepth : int
        Maximum sample tree depth control parameter. Default = 10.
    seed : {int32, None}
        Random seed.
    permute : bool, optional
        If True, permute the samples.
    init_method : {'stable', 0, 'random'}
        String determining the initialisation method for each chain. Default =
        'stable'.

    Returns
    -------
    Emulator : apricot.emulator.Emulator instance
        Emulator fit to data.
    """
    opts = {
        'jitter' : jitter,
        'fit_options' : fit_options,
        'samples' : samples,
        'thin' : thin,
        'chains' : chains,
        'adapt_delta' : adapt_delta,
        'max_treedepth' : max_treedepth,
        'seed' : seed,
        'permute' : permute,
        'init_method' : init_method,
    }
    samples, info = interface.hmc(x, y, **opts)
    hyperparameters = glue.hmc_glue(interface, samples, info)
    return Emulator(x, y, hyperparameters, info, jitter=jitter)

def _fit_mle(interface, x, y):
    """ Optimise log likelihood of the hyperparameters for the provided model.

    Parameters
    ----------
    model : apricot.Interface instance
        Interface to the desired Stan model.
    x : ndarray
        (n,d) array with each row representing a sample point in d-dimensional
        space.
    y : ndarray
        (n,) array of responses corresponding to each row of x.

    Returns
    -------
    Emulator : apricot.emulator.Emulator instance
    """
    result, info = interface.mle(x, y)
    hyperparameters = glue.mle_glue(interface, result, info)
    return Emulator(x, y, hyperparameters, info)
