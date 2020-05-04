"""
Code for the Gaussian Process Emulator class and associated utilities.

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import Mapping, Callable, Optional, Dict, Any, Tuple
from functools import wraps
import numpy as np  # type: ignore
from apricot.core import gp_internal  # pylint: disable=E0611
from apricot.core import sampling
from apricot.core import optimisation
from apricot.core import utils
from apricot.core import exceptions
from apricot.core import visualisation
from apricot.core.models import parse
from apricot.core.models import type_aliases as ta


def format_input(
        x_star: np.ndarray,
        index_dimension: int
) -> np.ndarray:
    """ Format inputs.

    Ensures arrays are correctly shaped and F-ordered before passing to the
    internal GP methods to prevent repeated copying / c++ runtime errors.

    Parameters
    ----------
    x_star : ndarray
        Unformatted array. Raw (n, index_dimension) array of n points from
        the index.
    index_dimension : int
        Dimension of index.

    Returns
    -------
    x_star_f : ndarray
        Formatted array. Strictly of shape (n, index_dimension) and F-ordered.

    Raises
    ------
    exceptions.ShapeError : if input array shape is not compatible with the
        model.
    """
    x_star = np.atleast_1d(x_star)
    if x_star.ndim == 1:
        if index_dimension == 1:
            x_star_f = x_star.reshape(-1, 1, order='F')
        elif x_star.shape[0] == index_dimension:
            x_star_f = x_star.reshape(1, -1, order='F')
        else:
            raise exceptions.ShapeError('x_star', 'n', 1, x_star.shape)
    elif x_star.ndim == 2:
        if x_star.shape[1] != index_dimension:
            raise exceptions.ShapeError(
                'x_star', 'n', index_dimension, x_star.shape)
        x_star_f = utils.force_f_array(x_star)
    else:
        x_star_s = x_star.squeeze()
        if x_star_s.ndim == 2:
            return format_input(x_star_s, index_dimension)
        raise exceptions.ShapeError(
            'x_star', 'n', index_dimension, x_star.shape)
    return x_star_f


def defined_on_index(method: Callable) -> Callable:
    """ Decorator for methods accepting arrays of points from the index.

    Applies format_inputs to arrays passed to the wrapped method, ensuring
    arrays are correctly shaped and F-ordered before passing them to the
    decorated methods.

    Parameters
    ----------
    method : Emulator method
        Method bound to an Emulator instance accepting an (n,d) array x_star
        as its first argument.

    Returns
    -------
    wrapped_method : Emulator method
        Original method wrapped with defined_on_index, such that inputs are
        always correctly shaped.
    """
    @wraps(method)
    def wrapper(inst, x_star, *tail, **kwargs):
        x_star_f = format_input(x_star, inst.index_dimension)
        return method(inst, x_star_f, *tail, **kwargs)
    return wrapper


def assign_internal_gp(kernel_type: str) -> ta.InternalGp:
    """ Assign internal GP based on requested kernel """
    assign: Mapping[str, ta.InternalGp] = {
        'eq': gp_internal.GpEqKernel,
        'eq_flat': gp_internal.GpEqKernel,
        'm52': gp_internal.GpM52Kernel,
        'm32': gp_internal.GpM32Kernel,
        'rq': gp_internal.GpRqKernel
    }
    return assign[parse.parse_kernel(kernel_type)]


class Emulator:

    """ Gaussian Process Emulator.

    User-facing interface to a compiled GP emulator.

    Attributes
    ----------
    x_data: ndarray
        (n,d) array of n sample points in d-dimensional space
    y_data: ndarray
        (n,) array of n sample responses, corresponding to the rows of x
    n_samples: int
        Number of observations
    n_hyperparameters: int
        Number of hyperparameter samples
    indeX_dimension: int
        Dimension of observations
    hyperparameters: dict
        Dictionary of model hyperparameters
    kernel_type: {'eq', 'm52', 'm32', 'rq'}, optional
        String designating the covariance kernel type:
        * 'eq' Exponentiated quadratic kernel
        * 'm52' Matern kernel with nu=5/2
        * 'm32' Matern kernel with nu=3/2
        * 'rq' Rational quadratic kernel
    mean_function_type: {'zero', None}, optional
        String designating the mean function type. Default = 'zero'
    info : dict
        Dictionary of diagnostic fit information
    m : int
        Number of hyperparameter samples

    Methods
    -------
    __call__(x_star)
        Posterior (predictive) expectation, integrated over hyperparameters
    expectation(x_star)
        Posterior (predictive) expectation (verbose wrapper for __call__),
        integrated over hyperparameters
    marginals(x_star)
        Posterior (predictive) marginal distributions at x_star for all m
        hyperparameters
    posterior(x_star)
        Posterior (predictive) joint distribution at x_star for all m
        hyperparameters
    loo_cv()
        Analytical leave-one-out cross validation scores for the n training
        sample points.
    ei(x_star)
        (Negative) expected improvement acquisition function, integrated over
        hyperparameters
    px(x_star)
        (Negative) posterior predictive variance ("pure exploration")
        acquisition function, integrated over hyperparameters
    ucb(x_star, beta)
        (Negative) upper confidence bound acquisition function, integrated over
        hyperparameters. This is a minimiser, so is strictly a "lower"
        confidence bound acquisition function.
    entropy(x_star)
        Differential entropy of predictive distribution at x_star, integrated
        over hyperparamters.
    optimisation.optimise(mode='min', initial_guess=None, grid=None,
        grid_size=None, grid_method='lhs', grid_options=None, seed=None)
        Numerical optimisation (min or max) of posterior expectation.
    next_ei(initial_guess=None, grid=None, grid_size=None, grid_method='lhs',
        grid_options=None, seed=None)
        Numerical optimisation of expected improvement.
    next_px(initial_guess=None, grid=None, grid_size=None, grid_method='lhs',
        grid_options=None, seed=None)
        Numerical optimisation of posterior predictive variance.
    next_ucb(beta, initial_guess=None, grid=None, grid_size=None,
        grid_method='lhs', grid_options=None, seed=None)
        Numerical optimisation of upper confidence bound.
    sobol1(n=1000, method='sobol', seed=None)
        First order Sobol indices
    plot_parameter(parameter)
        Trace plot of hyperparameter with name parameter.
    plot_divergences()
        Parallel co-ordinates plot of model hyperparameters. Highlights any
        divergent sampler transitions.

    Private Attributes
    ------------------
    _x: ndarray
        (n,d) array of n sample points in d dimensional space
    _y: ndarray
        (n,) array of n sample responses, corresponding to the rows in _x
    _n:
    _d:
    _m:
    _gp :
        (Internal) c++ GP object.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            hyperparameters: ta.Hyperparameters,
            info: Optional[Dict[str, Any]] = None,
            kernel_type: str = 'eq',
            mean_function_type: ta.MeanFunction = 'zero',
            jitter: float = 1e-10,
    ) -> None:
        """ Gaussian Process emulator.

        Parameters
        ----------
        x_data : ndarray
            (n, d) array of sample points.
        y_data : ndarray
            (n,) array of function responses.
        hyperparameters : dict
            Dictionary containing kernel hyperparameters.
        info : dict, optional
            Dictionary containing model fit information.
        kernel_type : {'eq', 'm52', 'm32', 'rq'}, optional
            String designating the kernel type. Default = 'eq'.
        mean_function_type : {'zero', None}, optional
            String designating the mean function type. Default = 'zero'.
        jitter : float, optional
            Magnitude of stability jitter. This is a standard deviation: supply
            the square root if designating a variance. Default = 1e-10.
        """

        # pylint: disable=too-many-arguments

        self._x = utils.force_f_array(x_data)
        self._y = utils.force_f_array(y_data)

        self.kernel_type = kernel_type
        self.mean_type = mean_function_type
        self.hyperparameters = hyperparameters

        # ----------------------------------------------------------------------
        # TODO: refactor
        try:
            self._amp = hyperparameters['amp']
            self._ls = hyperparameters['ls']
            self._sigma = hyperparameters['sigma']
            if self.kernel_type == 'rq':
                self._kappa = hyperparameters['kappa']
        except KeyError as missing_key:
            raise exceptions.MissingParameterError(str(missing_key)) from None
        self._jitter = jitter
        internal = assign_internal_gp(self.kernel_type)
        if self.kernel_type == 'rq':
            args = (
                self._x,
                self._y,
                self._amp,
                self._ls,
                self._kappa,
                self._sigma,
                self._jitter,
            )  # type: ignore
        else:
            # for some reason mypy wants the type signature of the internal rq
            # kernel here and is expecting kappa to be present -- I think this
            # warning can probably be ignored.
            args = (
                self._x,
                self._y,
                self._amp,
                self._ls,
                self._sigma,
                self._jitter,
            )  # type: ignore
        self._gp = internal(*args)  # type: ignore
        # ----------------------------------------------------------------------

        self._n, self._d = x_data.shape
        self._m = self._amp.shape[0]

        # if number of hyperparameter samples is 1 we can assume they were
        # optimised
        if info is None:
            info = {}
            if self._m > 1:
                info['method'] = 'hmc'
            else:
                # this could also be MLE as of 0.93
                info['method'] = 'map'
        self.info = info

    @property
    def x_data(self) -> np.ndarray:
        """ (n,d) array of sample points. """
        return self._x

    @property
    def y_data(self) -> np.ndarray:
        """ (n,) array of sample responses. """
        return self._y

    @property
    def index_dimension(self) -> int:
        """ Index dimension. """
        return self._d

    @property
    def n_samples(self) -> int:
        """ Number of sample points """
        return self._n

    @property
    def n_hyperparameters(self) -> np.ndarray:
        """ Number of hyperparameter samples """
        return self._m

    @defined_on_index
    def __call__(self, x_star: np.ndarray) -> np.ndarray:
        """ Posterior expectation.

        The posterior expectation of the emulator, integrated over the model
        hyperparameters.

        Parameters
        ----------
        x_star : ndarray
            (n,d) array of points at which to compute the posterior expectation

        Returns
        -------
        expectation : ndarray
            (n,) array corresponding to the posterior expectation at x_star

        Notes
        -----
        Identical to self.expectation(x_star)
        """
        return self._gp.E(x_star)

    def expectation(self, x_star: np.ndarray) -> np.ndarray:
        """Posterior expectation

        The posterior expectation of the emulator, integrated over the model
        hyperparameters.

        Parameters
        ----------
        x_star : ndarray
            (n,d) array of points at which to compute the posterior expectation

        Returns
        -------
        expectation : ndarray
            (n,) array corresponding to the posterior expectation at x_star

        Notes
        -----
        Just an explicit wrapper for self.__call__
        """
        return self.__call__(x_star)

    @defined_on_index
    def marginals(
            self,
            x_star: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """"Predictive marginals

        Marginal predictive distributions corresponding to each of the m
        hyperparameter samples at the points described by x_star.

        Parameters
        ----------
        x_star : ndarray
            (n, d) array of points at which to compute the predictive marginals

        Returns
        -------
        means : ndarray
            (n, m) array of means, corresponding to the mean at each of the n
            points in x_star for each of the m hyperparameter samples
        variances : ndarray
            (n, m) array of variances, corresponding to the variance at each of
            the n points in x_star for each of the m hyperparameter samples
        """
        return self._gp.marginals(x_star)

    @defined_on_index
    def posterior(
            self,
            x_star: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive distribution

        The (joint) predictive distribution of the model corresponding to each
        hyperparameter sample at the points described by x_star.

        Parameters
        ----------
        x_star : ndarray
            (n,d) array of points at which to compute the joint predictive
            distribution

        Returns
        -------
        means : ndarray
            (n,m) array of means, corresponding to the mean at each of the n
            points in x_star for each of the m hyperparameter samples
        covariance_matrices : ndarray
            (n,n,m) array consisting of m covariances matrices of size (n,n),
            describing the joint distribution over the n points in x_star for
            each of the m hyperparameter samples
        """
        return self._gp.posterior(x_star)

    def loo_cv(self) -> np.ndarray:
        """Leave-One-Out Cross Validation Scores

        Returns the analytical log predictive densities at each of the training
        samples using the method of Sundararajan & Keerthi [1]_.

        Returns
        -------
        cv_scores : ndarray

        References
        ----------
        [1] Sundararajan, S. and Keerthi, S.S., 2000. Predictive approaches for
        choosing hyperparameters in Gaussian processes. In Advances in neural
        information processing systems (pp. 631-637).

        Notes
        -----
        The method only delivers valid results for zero-mean Emulators.
        """
        return self._gp.loo_cv()

    @defined_on_index
    def expected_improvement(self, x_star: np.ndarray) -> np.ndarray:
        """Expected improvement acquisition function

        The (negative) Expected Improvement (EI) acquisition function of
        Mockus [1]_, integrated over the model hyperparameters.

        This implementation is negative both in the sense that it seeks an
        improvement in the *minimum* value of the target function *and* that
        the acquisition function itself is designed to be minimised.

        Parameters
        ----------
        x_star : ndarray
            (n,d) array of points at which the expected improvement should be
            evaluated

        Returns
        -------
        expected_improvement : ndarray
            (n,) array containing the expected improvement at each of the n
            points in x_star, integrated over the hyperparameter samples

        See Also
        --------
        next_ei

        References
        ----------
        [1] Mockus, J., Tiesis, V. and Zilinskas, A., 1978. The application of
        bayesian methods for seeking the extremum. vol. 2.

        """
        return self._gp.ei(x_star)

    @defined_on_index
    def pure_exploration(self, x_star: np.ndarray) -> np.ndarray:
        """Pure exploration acquisition function

        The Pure eXploration (PX) acquisition function is equivalent to
        the (negative) predictive marginal variance integrated over the model
        hyperparameters.

        This implementation of PX is negative, and the point of maximum
        posterior marginal variance lies at the point in the index which
        minimises the PX function.

        Parameters
        ----------
        x_star : ndarray
            (n,d) array of points at which the negative predictive variance
            should be evaluated

        Returns
        -------
        px : ndarray
            (n,) array containing the negative predictive variance at each of
            the n points in x_star, integrated over the hyperparameter samples

        See Also
        --------
        next_px
        """
        return self._gp.px(x_star)

    @defined_on_index
    def upper_confidence_bound(
            self,
            x_star: np.ndarray,
            beta: float
    ) -> np.ndarray:
        """Upper confidence bound acquisition function

        The Upper Confidence Bound (UCB) acquisition function of
        Srinivas et. al. [1]_, integrated over the model hyperparameters.

        This implementation is negative both in the sense that it seeks an
        improvement in the *minimum* value of the target function *and* that
        the acquisition function itself is to be **minimised**.

        Parameters
        ----------
        x_star : ndarray
            (n,d) array of points at which the upper confidence bound function
            should be evaluated
        beta : float
             Parameter beta for the upper confidence bound acquisition function

        Returns
        -------
        ucb : ndarray
            (n,) array containing the upper confidence bound function at each
            of the n points in x_star, integrated over the hyperparameter
            samples.

        Notes
        -----
        Strictly speaking this is a "lower confidence bound" acquisition
        function, since the acquisition function seeks the minimum.

        See Also
        --------
        next_ucb

        References
        ----------
        [1] Srinivas, N., Krause, A., Kakade, S.M. and Seeger, M., 2009.
        Gaussian process optimization in the bandit setting: No regret and
        experimental design. arXiv preprint arXiv:0912.3995.
        """
        return self._gp.ucb(x_star, beta)

    @defined_on_index
    def entropy(self, x_star: np.ndarray) -> np.ndarray:
        """Differential entropy

        Compute the differential entropy of the posterior (joint) distribution
        at the points described by x_star.

        Parameters
        ----------
        x_star : ndarray
            (n,d) array of points at which the differential entropy of the
            posterior distribution should be evaluated

        Returns
        -------
        H : float
            Differential entropy
        """
        return self._gp.entropy(x_star)

    def optimise(  # pylint: disable=too-many-arguments
            self,
            mode: str = 'min',
            initial_guess: Optional[np.ndarray] = None,
            grid: Optional[np.ndarray] = None,
            grid_size: Optional[int] = None,
            grid_method: str = 'lhs',
            grid_options: Optional[Mapping[str, Any]] = None,
            seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """ Global Min/Max of the posterior expectation.

        Use numerical optimisation to estimate the global minimum or maximum of
        the expectation of the predictive distribution.

        Parameters
        ----------
        mode : {'min', 'max'}, optional
            If mode is 'min', attempt to find the global minimum of the
            predictive distribution. If mode is 'max', attempt to find the
            global maximum. Default = 'min'.
        initial_guess : {None, ndarray}, optional
            (d,) array representing the initialisation point for the
            optimisation routine. Since a gradient based optimiser is used,
            a good starting point helps to avoid convergence to a local
            optimum. If None, a preliminary grid search will be performed
            to determine a suitable initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify initial_guess. If None, a grid will be
            generated. Default = None.
        grid_size : {None, int}, optional
            Integer specifying the number of points to use in the preliminary
            grid search. If None, use 100*d. Default = None.
        grid_method : str, optional
            String specifying the experimental design strategy used to
            construct the grid of points used to conduct the grid search.
            Default = 'lhs'.
        grid_options : dict, optional
            Dictionary of additional options to pass to grid_method. Default
            = None.
        seed : {None, int32}
            Random seed. Used for preliminary grid generation. Default = None.

        Returns
        -------
        result : dict
            Optimisation results.
        """
        _mode = mode.lower()
        if _mode == 'min':
            obj = self._gp.E
            obj_jac = self._gp.E_jac
        elif _mode == 'max':

            def obj(x_star):
                return -self._gp.E(x_star)

            def make_both_negative(f_x, f_x_jac):
                return (-f_x, -f_x_jac)

            def obj_jac(x_star):
                return make_both_negative(*self._gp.E_jac(x_star))

        else:
            raise ValueError("Mode must be either 'min' or 'max'.")

        opts: Mapping[str, Any] = {
            'x0': initial_guess,
            'grid': grid,
            'grid_size': grid_size,
            'grid_method': grid_method,
            'grid_options': grid_options,
            'seed': seed,
        }
        result = optimisation.optimise(obj, obj_jac, self._d, **opts)

        if _mode == 'max':
            result['xprime'] = -result['xprime']

        return result

    def next_ei(  # pylint: disable=too-many-arguments
            self,
            initial_guess: Optional[np.ndarray] = None,
            grid: Optional[np.ndarray] = None,
            grid_size: Optional[int] = None,
            grid_method: str = 'lhs',
            grid_options: Optional[Mapping[str, Any]] = None,
            seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """ Get the next point using expected improvement.

        Use numerical optimisation to estimate the global minimum of the
        (negative) Expected Improvement (EI) acquisition function [1]_, hence
        determining the point in the index with the highest EI.

        Parameters
        ----------
        initial_guess : {None, ndarray}, optional
            (d,) array representing the initialisation point for the
            optimisation routine. Since a gradient based optimisation.optimiser
            is used, a good starting point helps to avoid convergence to a
            local optimum. If None, a preliminary grid search will be performed
            to determine a suitable initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify initial_guess. If None, a grid will be
            generated. Default = None.
        grid_size : {None, int}, optional
            Integer specifying the number of points to use in the preliminary
            grid search. If None, use 100*d. Default = None.
        grid_method : str, optional
            String specifying the experimental design strategy used to
            construct the grid of points used to conduct the grid search. If
            None, defaults to 'lhs'. Default = None.
        grid_options : dict, optional
            Dictionary of additional options to pass to grid_method. Default
            = None.
        seed : {None, int32}
            Random seed. Used for preliminary grid generation. Default = None.

        Returns
        -------
        result : dict
            Optimisation results.

        See Also
        --------
        ei

        References
        ----------
        [1] Mockus, J., Tiesis, V. and Zilinskas, A., 1978. The application of
        bayesian methods for seeking the extremum. vol. 2.
        """

        func = self._gp.ei
        func_and_jac = self._gp.ei_jac
        opts: Mapping[str, Any] = {
            'x0': initial_guess,
            'grid': grid,
            'grid_size': grid_size,
            'grid_method': grid_method,
            'grid_options': grid_options,
            'seed': seed,
        }
        return optimisation.optimise(func, func_and_jac, self._d, **opts)

    def next_px(  # pylint: disable=too-many-arguments
            self,
            initial_guess: Optional[np.ndarray] = None,
            grid: Optional[np.ndarray] = None,
            grid_size: Optional[int] = None,
            grid_method: str = 'lhs',
            grid_options: Optional[Mapping[str, Any]] = None,
            seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """ Get the next point using pure exploration.

        Use numerical optimisation to estimate the global minimum of the
        (negative) predictive variance (Pure eXploration; PX) acquisition
        function, hence determining the point in the index with the highest
        PX.

        Parameters
        ----------
        initial_guess : {None, ndarray}, optional
            (d,) array representing the initialisation point for the
            optimisation routine. Since a gradient based optimiser is used, a
            good starting point helps to avoid convergence to a local optimum.
            If None, a preliminary grid search will be performed to determine a
            suitable initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify initial_guess. If None, a grid will be
            generated. Default = None.
        grid_size : {None, int}, optional
            Integer specifying the number of points to use in the preliminary
            grid search. If None, use 100*d. Default = None.
        grid_method : str, optional
            String specifying the experimental design strategy used to
            construct the grid of points used to conduct the grid search. If
            None, defaults to 'lhs'. Default = None.
        grid_options : dict, optional
            Dictionary of additional options to pass to grid_method. Default
            = None.
        seed : {None, int32}
            Random seed. Used for preliminary grid generation. Default = None.

        Returns
        -------
        result : dict
            Optimisation results.

        See Also
        --------
        px
        """

        func = self._gp.px
        func_and_jac = self._gp.px_jac
        opts: Mapping[str, Any] = {
            'x0': initial_guess,
            'grid': grid,
            'grid_size': grid_size,
            'grid_method': grid_method,
            'grid_options': grid_options,
            'seed': seed,
        }
        return optimisation.optimise(func, func_and_jac, self._d, **opts)

    def next_ucb(  # pylint: disable=too-many-arguments
            self,
            beta: float,
            initial_guess: Optional[np.ndarray] = None,
            grid: Optional[np.ndarray] = None,
            grid_size: Optional[int] = None,
            grid_method: str = 'lhs',
            grid_options: Optional[Mapping[str, Any]] = None,
            seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """ Get next point using upper confidence bound.

        Use numerical optimisation to estimate the global minimum of the
        (negative) Upper Confidence Bound (UCB) acquisition function [1]_,
        hence determining the point in the index with the highest UCB.

        Parameters
        ----------
        beta : float
            Parameter beta for the upper confidence bound acquisition function
        initial_guess : {None, ndarray}, optional
            (d,) array representing the initialisation point for the
            optimisation routine. Since a gradient based optimiser is used, a
            good starting point helps to avoid convergence to a local optimum.
            If None, a preliminary grid search will be performed to determine a
            suitable initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify a suitable start point. If None, a grid will be
            generated. Default = None.
        grid_size : {None, int}, optional
            Integer specifying the number of points to use in the preliminary
            grid search. If None, use 100*d. Default = None.
        grid_method : str, optional
            String specifying the experimental design strategy used to
            construct the grid of points used to conduct the grid search. If
            None, defaults to 'lhs'. Default = None.
        grid_options : dict, optional
            Dictionary of additional options to pass to grid_method. Default
            = None.
        seed : {None, int32}
            Random seed. Used for preliminary grid generation. Default = None.

        Returns
        -------
        result : dict
            Optimisation results.

        See Also
        --------
        ucb

        References
        ----------
        [1] Srinivas, N., Krause, A., Kakade, S.M. and Seeger, M., 2009.
        Gaussian process optimization in the bandit setting: No regret and
        experimental design. arXiv preprint arXiv:0912.3995.
        """

        #  closures defines objective function and jacobian with assigned beta
        def func(x_star):
            return self._gp.ucb(x_star, float(beta))

        def func_and_jac(x_star):
            return self._gp.ucb_jac(x_star, float(beta))

        opts: Mapping[str, Any] = {
            'x0': initial_guess,
            'grid': grid,
            'grid_size': grid_size,
            'grid_method': grid_method,
            'grid_options': grid_options,
            'seed': seed,
        }
        return optimisation.optimise(func, func_and_jac, self._d, **opts)

    def sobol1(
            self,
            sample_size: int = 1000,
            method: str = 'sobol',
            seed: Optional[int] = None,
    ) -> np.ndarray:
        """ Calculate first order Sobol indices for the emulator.

        Approximates first order Sobol indices for the emulator using the
        Monte-Carlo simulation method as described by [1]_.

        Parameters
        ----------
        n : int, optional
            Number of queries. A total of n*(d+2) expectations are computed.
            Default = 1000.
        method : str, optional
            Sampling method. Default = 'sobol'.
        seed : {None, int32}
            Random seed. Used for sample grid generation. Default = None.

        Returns
        -------
        sobol_indices : ndarray

        References
        ----------
        [1] Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto,
        M. and Tarantola, S., 2010. Variance based sensitivity analysis of
        model output. Design and estimator for the total sensitivity index.
        Computer Physics Communications, 181(2), pp.259-270.
        """

        x_arr = sampling.sample_hypercube(
            sample_size,
            2*self._d,
            method=method,
            seed=seed
        )
        a_arr = x_arr[:, :self._d]
        b_arr = x_arr[:, self._d:]
        f_a = self.expectation(a_arr)
        f_b = self.expectation(b_arr)
        var_local = np.empty(self._d)
        f_all = [f_a, f_b]
        for i in range(self._d):
            ab_arr = a_arr.copy()
            ab_arr[:, i] = b_arr[:, i]
            f_i = self.expectation(ab_arr)
            var_local[i] = np.mean(f_b * (f_i - f_a))
            f_all.append(f_i)
        var_total = np.var(np.array(f_all).ravel())
        return var_local / var_total

    def plot_parameter(self, param_name: str) -> None:
        """ Trace plot of hyperparameter with param_name"""
        if self.info['method'] == 'hmc':
            visualisation.plot_parameter(
                self.hyperparameters,
                param_name,
                self.info
            )
        raise RuntimeError('Method is only valid for models fit using HMC.')

    def plot_divergences(self) -> None:
        """ Parallel co-ordinates plot of sampler behaviour

        Highlights any divergent transitions.
        """
        if self.info['method'] == 'hmc':
            visualisation.plot_divergences(
                self.hyperparameters,
                self.info
            )
        raise RuntimeError('Method is only valid for models fit using HMC.')
