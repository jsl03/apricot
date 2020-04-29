# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
import typing
import numpy as np
from apricot.core import emulator
from apricot.core.models import interface, glue, mle
from apricot.core.logger import get_logger


def fit(
        x: np.ndarray,
        y: np.ndarray,
        kernel: str = 'eq',
        mean: str = 'zero',
        noise: typing.Union[str, float] = 'zero',
        method: str = 'hmc',
        **kwargs,
) -> emulator.Emulator:
    """ Fit a Gaussian Process emulator to data.

    Given inputs, x, and outputs, y, infer the hyperparameters of a GP
    emulator with kernel 'kernel', mean function 'mean', and noise type
    'noise'.

    Valid fit methods are 'hmc', for Hamiltonian Monte-Carlo sampling from the
    posterior distribution of the model hyperparameters, 'map' for maximum
    a-posteriori probability estimation, and 'mle' for maximimum marginal
    likelihood estimation.

    'map' and 'mle' are both modal approximations that use some from of
    numerical optimisation to identify the hyperparameters with the highest
    posterior probability (crucially, including a prior) and marginal
    likelihood, respectively.

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
        * 'm52' Matern kernel with smoothness parameter nu=5/2.
        * 'm32' Matern kernel with smoothness parameter nu=3/2.
        * 'rq' Rational quadratic kernel.
    mean : {None}
        String describing the desired mean function.
        * {None, 'zero', 0} Zero mean function
    noise : {None}
        String describing the desired noise model.
        * {None, 'zero', 0} Zero (noiseless) model.
    method : {'hmc', 'map', mle'}
        string describing the desired fit method:
        * 'hmc' Hamiltonian Monte-Carlo. Uses Stan's implementation of the
            No-U-Turn Sampler (NUTS).
        * 'map' Maximum a-posteriori estimation (modal approximation). Maximise
            the posterior probability of the hyperparameters using numerical
            optimisation. Uses one of Stan's gradient based optimisers.
        * 'mle' Maximum Log Likelihood Estimation. Maximise the log *marginal*
            likelihood using numerical optimisation. Currently only valid with
            kernel = 'eq'.
    init_method : {None, 'stable', 'zero', 'random', dict}
        Determines initialisation method. If method = 'hmc', fit_method
        determines the start points for each of the requested sample chains.
        If method = 'map', fit_method determines the start point for the first
        optimisation only since subsequent restarts are initialised randomly.
        Ignored if method = 'mle'. Defaults to = 'stable'.
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
    seed : {int32, None}
        Random seed.
    samples : int, optional
        Valid only if fit_method = 'hmc'. Number of samples to draw from the
        posterior (accounting for the number of chains, warmup and thinning).
        Default = 2000.
    thin : int
        Valid only if fit_method = 'hmc'. If > 1, keep only every thin samples.
        Default = 1.
    chains : int
        Valid only if fit_method = 'hmc'. The number of independent chains to
        draw samples from. Default = 4.
    adapt_delta : float < 1
        Valid only if fit_method = 'hmc'. adapt_delta control parameter for the
        sampler. Default = 0.8.
    max_treedepth : int
        Valid only if fit_method = 'hmc'. Maximum sample tree depth control
        parameter. Default = 10.
    permute : bool, optional
        Valid only if method = 'hmc'. If True, permute the samples.
        Default = True.
    algorithm : str, optional
        Valid only if fit_method = 'map'. String specifying which of Stan's
        gradient based optimisation algorithms to use. Default = 'Newton'.
    restarts : int, optional
        Valid only if fit_method = 'map'. The number of restarts to use.
        The optimisation will be repeated this many times and the
        hyperparameters with the highest log-likelihood will be returned.
        Default=10.
    max_iter : int, optional
        Valid only if fit_method = 'map'. The maximum allowable number of
        iterations for the chosen optimisation algorithm. Default = 250.

    Returns
    -------
    emulator_instance : instance of apricot.emulator.Emulator
        Gaussian Process emulator.

    """

    interface_instance = interface.Interface(kernel, mean, noise)

    if method.lower() == 'hmc':
        return _fit_hmc(interface_instance, x, y, **kwargs)
    elif method.lower() == 'map':
        return _fit_map(interface_instance, x, y, **kwargs)
    elif method.lower() == 'mle':
        jitter = kwargs['jitter'] if 'jitter' in kwargs else 1e-10
        return _fit_mle(interface_instance, x, y, jitter)
    else:
        raise ValueError("Unrecognised fit method: '{0}'.".format(method))


def _fit_hmc(
        interface_instance: interface.Interface,
        x: np.ndarray,
        y: np.ndarray,
        jitter: float = 1e-10,
        fit_options: typing.Optional[dict] = None,
        samples: int = 4000,
        thin: int = 1,
        chains: int = 4,
        adapt_delta: float = 0.8,
        max_treedepth: int = 10,
        seed: typing.Optional[int] = None,
        permute: bool = True,
        init_method: typing.Union[dict, str, int] = 'stable',
) -> emulator.Emulator:
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
        Specify either a list of 'linear' or 'nonlinear' for each input
        dimension, or a single string to apply the supplied option to all input
        dimensions.
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
    init_method : {'stable', 'zero', 'random', dict}
        Determines initialisation method for each chain. Default = 'stable'.
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
    Emulator : apricot.emulator.Emulator instance
        Emulator fit to data.
    """
    samples, info = interface_instance.hmc(
        x,
        y,
        jitter=jitter,
        fit_options=fit_options,
        samples=samples,
        thin=thin,
        chains=chains,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        permute=permute,
        init_method=init_method
    )
    hyperparameters = glue.hmc_glue(interface_instance, samples, info)
    emulator_instance = emulator.Emulator(
        x,
        y,
        hyperparameters,
        info=info,
        kernel_type=interface_instance.kernel_type,
        mean_function_type=interface_instance.mean_function_type,
        jitter=jitter
    )
    return emulator_instance


def _fit_map(
        interface_instance: interface.Interface,
        x: np.ndarray,
        y: np.ndarray,
        jitter: float = 1e-10,
        init_method: typing.Union[dict, str, int] = 'stable',
        algorithm: str = 'Newton',
        restarts: int = 5,
        max_iter: int = 250,
        seed: typing.Optional[int] = None,
) -> emulator.Emulator:
    """ Optimise the posterior probability of the hyperparameters for the
    provided model.

    Parameters
    ----------
    model : apricot.Interface instance
        Interface to the desired Stan model.
    x : ndarray
        (n,d) array with each row representing a sample point in d-dimensional
        space.
    y : ndarray
        (n,) array of responses corresponding to each row of x.
    jitter : float, optional
        Magnitude of stability jitter. Default = 1e-10.
    init_method : {'stable', 'zero', 'random', dict}
        Determines initialisation method. Note that for restarts > 1,
        each optimisation routine after the first will be initialised with
        init_method = 'random'. Default = 'stable'.
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
        Random seed.

    Returns
    -------
    Emulator : apricot.emulator.Emulator instance
    """
    result, info = interface_instance.map(
        x,
        y,
        jitter,
        init_method=init_method,
        algorithm=algorithm,
        restarts=restarts,
        max_iter=max_iter,
        seed=seed
    )
    hyperparameters = glue.map_glue(interface_instance, result, info)
    emulator_instance = emulator.Emulator(
        x,
        y,
        hyperparameters,
        info=info,
        kernel_type=interface_instance.kernel_type,
        mean_function_type=interface_instance.mean_function_type,
        jitter=jitter
    )
    return emulator_instance


def _fit_mle(interface_instance, x, y, jitter):
    """ Optimise the log marginal likelihood for the provided model """
    hyperparameters, info = mle.run_mle(interface_instance, x, y, jitter)
    emulator_instance = emulator.Emulator(
        x,
        y,
        hyperparameters,
        info=info,
        kernel_type=interface_instance.kernel_type,
        mean_function_type=interface_instance.mean_function_type,
        jitter=jitter
    )
    return emulator_instance
