"""
Methods for fitting Emulators to data.

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import Any, Optional, Union
import numpy as np  # type: ignore
from apricot.core import emulator
from apricot.core.models import interface
from apricot.core.models import glue
from apricot.core.models import mle
from apricot.core.models import cv
from apricot.core.models import type_aliases as ta
from apricot.core.logger import get_logger


LOGGER = get_logger()


def fit(  # pylint: disable=too-many-arguments
        x_data: np.ndarray,
        y_data: np.ndarray,
        kernel: str = 'eq',
        mean: str = 'zero',
        noise: Union[str, float] = 'zero',
        method: str = 'hmc',
        **kwargs: Any,
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
    x_data: ndarray
        (n, d) array with each row representing a sample point in d-dimensional
        space.
    y_data: ndarray
        (n,) array of responses corresponding to each row of x.
    kernel: {'expq'}
        String describing the desired kernel type:
        * 'expq' Exponentiated quadratic ("squared exponential").
        * 'm52' Matern kernel with smoothness parameter nu=5/2.
        * 'm32' Matern kernel with smoothness parameter nu=3/2.
        * 'rq' Rational quadratic kernel.
    mean: {None}
        String describing the desired mean function.
        * {None, 'zero', 0} Zero mean function
    noise: {None}
        String describing the desired noise model.
        * {None, 'zero', 0} Zero (noiseless) model.
    method: {'hmc', 'map', mle'}
        string describing the desired fit method:
        * 'hmc' Hamiltonian Monte-Carlo. Uses Stan's implementation of the
            No-U-Turn Sampler (NUTS).
        * 'map' Maximum a-posteriori estimation (modal approximation). Maximise
            the posterior probability of the hyperparameters using numerical
            optimisation. Uses one of Stan's gradient based optimisers.
        * 'mle' Maximum Log Likelihood Estimation. Maximise the log *marginal*
            likelihood using numerical optimisation. Currently only valid with
            kernel = 'eq'.
        * 'cv' Leave-one-out cross validated predictive log density
            optimisation.
    init_method: {None, 'stable', 'zero', 'random', dict}
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
        * 'zero': initialise all parameters from zero on the unconstrained
            support.
        * 'random': initialise all parameters randomly on their support.
        * 0: Equivalent to 'zero' (see above).
        * dict: A custom initialisation value for each of the model's
            parameters.
    seed: {int32, None}
        Random seed.
    samples: int, optional
        Valid only if fit_method = 'hmc'. Number of samples to draw from the
        posterior (accounting for the number of chains, warmup and thinning).
        Default = 2000.
    thin: int
        Valid only if fit_method = 'hmc'. If > 1, keep only every thin samples.
        Default = 1.
    chains: int
        Valid only if fit_method = 'hmc'. The number of independent chains to
        draw samples from. Default = 4.
    adapt_delta : float < 1
        Valid only if fit_method = 'hmc'. adapt_delta control parameter for the
        sampler. Default = 0.8.
    max_treedepth: int
        Valid only if fit_method = 'hmc'. Maximum sample tree depth control
        parameter. Default = 10.
    permute: bool, optional
        Valid only if method = 'hmc'. If True, permute the samples.
        Default = True.
    algorithm: str, optional
        Valid only if fit_method = 'map'. String specifying which of Stan's
        gradient based optimisation algorithms to use. Default = 'Newton'.
    restarts: int, optional
        Valid only if fit_method = 'map'. The number of restarts to use.
        The optimisation will be repeated this many times and the
        hyperparameters with the highest log-likelihood will be returned.
        Default=10.
    max_iter: int, optional
        Valid only if fit_method = 'map'. The maximum allowable number of
        iterations for the chosen optimisation algorithm. Default = 250.

    Returns
    -------
    emulator_instance: instance of apricot.emulator.Emulator
        Gaussian Process emulator.
    """

    interface_instance = interface.Interface(kernel, mean, noise)

    if method.lower() == 'hmc':
        return fit_hmc(interface_instance, x_data, y_data, **kwargs)

    if method.lower() == 'map':
        return fit_map(interface_instance, x_data, y_data, **kwargs)

    if method.lower() == 'mle':
        return fit_mle(interface_instance, x_data, y_data, **kwargs)

    if method.lower() == 'cv':
        return fit_cv(interface_instance, x_data, y_data, **kwargs)

    raise ValueError("Unrecognised fit method: '{0}'.".format(method))


def fit_hmc(  # pylint: disable=too-many-arguments, too-many-locals
        interface_instance: interface.Interface,
        x_data: np.ndarray,
        y_data: np.ndarray,
        jitter: float = 1e-10,
        ls_options: ta.LsPriorOptions = None,
        samples: int = 4000,
        thin: int = 1,
        chains: int = 4,
        adapt_delta: float = 0.8,
        max_treedepth: int = 10,
        seed: Optional[int] = None,
        permute: bool = True,
        init_method: ta.InitTypes = 'stable',
) -> emulator.Emulator:
    """ Run Stan's HMC algorithm for the provided model.

    Parameters
    ----------
    interface: apricot.Interface instance
        Interface to the desired Stan model.
    x_data: ndarray
        (n, d) array with each row representing a sample point in d-dimensional
        space.
    y_data: ndarray
        (n,) array of responses corresponding to each row of x.
    jitter: float, optional
        Stability jitter. Default = 1e-10.
    ls_options: string or list of string, optional
        Specify either a list of 'linear' or 'nonlinear' for each input
        dimension, or a single string to apply the supplied option to all input
        dimensions.
    samples: int, optional
        Number of samples to draw from the posterior (accounting for the number
        of chains, warmup and thinning). Default = 2000.
    thin: int
        If > 1, keep only every thin samples. Default = 1.
    chains: int
        The number of independent chains to draw samples from. Default = 4.
    adapt_delta: float < 1
        adapt_delta control parameter for the sampler. Default = 0.8.
    max_treedepth: int
        Maximum sample tree depth control parameter. Default = 10.
    seed: {int32, None}
        Random seed.
    permute: bool, optional
        If True, permute the samples.
    init_method: {'stable', 'zero', 'random', 0, dict}
        Determines initialisation method for each chain. Default = 'stable'.
        * 'stable': "stable" initialise parameters from data:
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
        * 'zero': initialise all parameters from zero on the unconstrained
            support.
        * 'random': initialise all parameters randomly on their support.
        * 0: Equivalent to 'zero' (see above).
        * dict: A custom initialisation value for each of the model's
            parameters.

    Returns
    -------
    Emulator: apricot.emulator.Emulator instance
        Emulator fit to data.
    """
    samples, info = interface_instance.hmc(
        x_data,
        y_data,
        jitter=jitter,
        ls_options=ls_options,
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
        x_data,
        y_data,
        hyperparameters,
        info=info,
        kernel_type=interface_instance.kernel_type,
        mean_function_type=interface_instance.mean_function_type,
        jitter=jitter
    )
    return emulator_instance


def fit_map(  # pylint: disable=too-many-arguments
        interface_instance: interface.Interface,
        x_data: np.ndarray,
        y_data: np.ndarray,
        jitter: float = 1e-10,
        init_method: ta.InitTypes = 'stable',
        algorithm: str = 'Newton',
        restarts: int = 5,
        max_iter: int = 250,
        seed: Optional[int] = None,
) -> emulator.Emulator:
    """ Optimise the posterior probability of the hyperparameters for the
    provided model.

    Parameters
    ----------
    interface_instance: apricot.Interface instance
        Interface to the desired Stan model.
    x: ndarray
        (n,d) array with each row representing a sample point in d-dimensional
        space.
    y: ndarray
        (n,) array of responses corresponding to each row of x.
    jitter: float, optional
        Magnitude of stability jitter. Default = 1e-10.
    init_method: {'stable', 'zero', 'random', dict}
        Determines initialisation method. Note that for restarts > 1,
        each optimisation routine after the first will be initialised with
        init_method = 'random'. Default = 'stable'.
        * 'stable': "stable" initialise parameters from data:
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
        * 'zero': initialise all parameters from zero on the unconstrained
            support.
        * 'random': initialise all parameters randomly on their support.
        * 0: Equivalent to 'zero' (see above).
        * dict: A custom initialisation value for each of the model's
            parameters.
    algorithm: str, optional
        String specifying which of Stan's gradient based optimisation
        algorithms to use. Default = 'Newton'.
    restarts: int, optional
        The number of restarts to use. The optimisation will be repeated
        this many times and the hyperparameters with the highest
        log-likelihood will be returned. restarts > 1 is not compatible with
        initialisation = 0. Default=10.
    max_iter: int, optional
        Maximum allowable number of iterations for the chosen optimisation
        algorithm. Default = 250.
    seed: {int32, None}
        Random seed.

    Returns
    -------
    Emulator: apricot.emulator.Emulator instance
    """
    result, info = interface_instance.map(
        x_data,
        y_data,
        jitter,
        init_method=init_method,
        algorithm=algorithm,
        restarts=restarts,
        max_iter=max_iter,
        seed=seed
    )
    hyperparameters = glue.map_glue(interface_instance, result)
    emulator_instance = emulator.Emulator(
        x_data,
        y_data,
        hyperparameters,
        info=info,
        kernel_type=interface_instance.kernel_type,
        mean_function_type=interface_instance.mean_function_type,
        jitter=jitter
    )
    return emulator_instance


def fit_mle(  # pylint: disable=too-many-arguments, too-many-locals
        interface_instance: interface.Interface,
        x_data: np.ndarray,
        y_data: np.ndarray,
        jitter: float = 1e-10,
        bounds: Optional[ta.Bounds] = None,
        ls_lower: float = 0.05,
        ls_upper: float = 1.0,
        sigma_lower: float = 0.0,
        sigma_upper: Optional[float] = None,
        amp_lower: float = 0.0,
        amp_upper: float = 1.0,
        grid_size: int = 5000,
        grid_method: str = 'sobol',
        grid_options: Optional[dict] = None,
        seed: Optional[int] = None,
        callback: Optional[ta.CallbackFunction] = None,
) -> emulator.Emulator:
    """ Identify model hyperparameters using maximum log marginal likelihood.

    Parameters
    ----------
    interface_instance: intstance of apricot.core.models.interface.Interface
        Interface to the model who's hyperparameters should be identified.
    x_data: ndarray
        (n, d) array of n sample points in d dimensions.
    y_data: ndarray
        (n) array of n responses corresponding to the rows of x.
    jitter: float, optional
        Magnitude of stability jitter to be added to the leading diagonal of
        the sample covariance matrix.
    bounds: List of tuple, optional
        List of (lower, upper) bounds for the preliminary grid search for each
        hyperparameter in the following order: signal amplitude (marginal
        standard deviation), (optionally) noise variance (if present),
        anisotropic lengthscales. For fixed noise models, this is:
        [amp, ls_1, ..., ls_d], and if inferring the noise standard deviation,
        this is [amp, xi, ls_1, ..., ls_d]. If not provided, the grid search
        bounds will be set to "sensible" defaults (see below). Default = None.
    ls_lower: float, optional
        Grid search lower bound for anisotropic lengthscales. Default = 0.05.
    ls_upper: float, optional
        Grid search upper bound for anisotropic lengthscales. Default = 1.
    sigma_lower: float, optional
        Grid search lower bound for noise standard deviation. Default = 0
    sigma_upper: {None, float}, optional
        Grid search upper bound for noise standard deviation. If None and
        infer_noise is True, sigma_upper is equal to 1/10 of the
        sample standard deviation for function observations, that is:
        sigma_upper = np.std(y) / 10. Default = None.
    amp_lower: float, optional
        Grid search lower bound for marginal standard deviation. Default = 0.
    amp_upper: float, optional
        Grid search upper bound for marginal standard deviation. Default = 1.
    grid_size: int, optional
        Number of points to use for the preliminary grid search. Default =
        5000. For large sample sample sizes, this can be reduced.
    grid_method: str, optional
        String specifying which method to use to generate the grid points.
        Valid options are compatible with apricot.sample_hypercube. Default =
        'sobol'.
    grid_options: dict, optional
        Additional options to pass to the chosen grid generating method.
        Default = None.
    seed: int32, optional
        Random seed. If not provided, one will be generated. Default = None.
    callback: ndarray -> Any
        Callback function. Recieves the parameter vector queried by the
        optimisation algorithm at each iteration. Used primarily for debugging.

    Returns
    -------
    Emulator : apricot.emulator.Emulator instance
    """
    hyperparameters, info = mle.run_mle(
        interface_instance,
        x_data,
        y_data,
        jitter=jitter,
        bounds=bounds,
        ls_lower=ls_lower,
        ls_upper=ls_upper,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        amp_lower=amp_lower,
        amp_upper=amp_upper,
        grid_size=grid_size,
        grid_method=grid_method,
        grid_options=grid_options,
        seed=seed,
        callback=callback
    )
    emulator_instance = emulator.Emulator(
        x_data,
        y_data,
        hyperparameters,
        info=info,
        kernel_type=interface_instance.kernel_type,
        mean_function_type=interface_instance.mean_function_type,
        jitter=jitter
    )
    return emulator_instance


def fit_cv(  # pylint: disable=too-many-arguments, too-many-locals
        interface_instance: interface.Interface,
        x_data: np.ndarray,
        y_data: np.ndarray,
        jitter: float = 1e-10,
        bounds: Optional[ta.Bounds] = None,
        ls_lower: float = 0.05,
        ls_upper: float = 1.0,
        sigma_lower: float = 0.0,
        sigma_upper: Optional[float] = None,
        amp_lower: float = 0.0,
        amp_upper: float = 1.0,
        grid_size: int = 5000,
        grid_method: str = 'sobol',
        grid_options: Optional[dict] = None,
        seed: Optional[int] = None,
        callback: Optional[ta.CallbackFunction] = None,
) -> emulator.Emulator:
    """ Identify model hyperparameters using maximum leave-one-out cross
    validated log predictive density.

    Parameters
    ----------
    interface_instance: intstance of apricot.core.models.interface.Interface
        Interface to the model who's hyperparameters should be identified.
    x_data: ndarray
        (n, d) array of n sample points in d dimensions.
    y_data: ndarray
        (n) array of n responses corresponding to the rows of x.
    jitter: float, optional
        Magnitude of stability jitter to be added to the leading diagonal of
        the sample covariance matrix.
    bounds: List of tuple, optional
        List of (lower, upper) bounds for the preliminary grid search for each
        hyperparameter in the following order: signal amplitude (marginal
        standard deviation), (optionally) noise variance (if present),
        anisotropic lengthscales. For fixed noise models, this is:
        [amp, ls_1, ..., ls_d], and if inferring the noise standard deviation,
        this is [amp, xi, ls_1, ..., ls_d]. If not provided, the grid search
        bounds will be set to "sensible" defaults (see below). Default = None.
    ls_lower: float, optional
        Grid search lower bound for anisotropic lengthscales. Default = 0.05.
    ls_upper: float, optional
        Grid search upper bound for anisotropic lengthscales. Default = 1.
    sigma_lower: float, optional
        Grid search lower bound for noise standard deviation. Default = 0
    sigma_upper: {None, float}, optional
        Grid search upper bound for noise standard deviation. If None and
        infer_noise is True, sigma_upper is equal to 1/10 of the
        sample standard deviation for function observations, that is:
        sigma_upper = np.std(y) / 10. Default = None.
    amp_lower: float, optional
        Grid search lower bound for marginal standard deviation. Default = 0.
    amp_upper: float, optional
        Grid search upper bound for marginal standard deviation. Default = 1.
    grid_size: int, optional
        Number of points to use for the preliminary grid search. Default =
        5000. For large sample sample sizes, this can be reduced.
    grid_method: str, optional
        String specifying which method to use to generate the grid points.
        Valid options are compatible with apricot.sample_hypercube. Default =
        'sobol'.
    grid_options: dict, optional
        Additional options to pass to the chosen grid generating method.
        Default = None.
    seed: int32, optional
        Random seed. If not provided, one will be generated. Default = None.
    callback: ndarray -> Any
        Callback function. Recieves the parameter vector queried by the
        optimisation algorithm at each iteration. Used primarily for debugging.

    Returns
    -------
    Emulator : apricot.emulator.Emulator instance
    """
    hyperparameters, info = cv.run_cv(
        interface_instance,
        x_data,
        y_data,
        jitter=jitter,
        bounds=bounds,
        ls_lower=ls_lower,
        ls_upper=ls_upper,
        sigma_lower=sigma_lower,
        sigma_upper=sigma_upper,
        amp_lower=amp_lower,
        amp_upper=amp_upper,
        grid_size=grid_size,
        grid_method=grid_method,
        grid_options=grid_options,
        seed=seed,
        callback=callback
    )
    emulator_instance = emulator.Emulator(
        x_data,
        y_data,
        hyperparameters,
        info=info,
        kernel_type=interface_instance.kernel_type,
        mean_function_type=interface_instance.mean_function_type,
        jitter=jitter
    )
    return emulator_instance
