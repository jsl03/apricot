import typing
from functools import wraps
import numpy as np

from apricot.core import gp_internal
from apricot.core import sampling
from apricot.core import optimisation
from apricot.core import utils
from apricot.core import exceptions
from apricot.core import visualisation


def _format_input(xstar: np.ndarray, d: int):
    """ Format inputs.

    Ensures arrays are correctly shaped and F-ordered before passing to the
    internal GP methods to prevent repeated copying / c++ runtime errors.

    Parameters
    ----------
    xstar : ndarray
        Unformatted array. Raw (n,d) array of n points from the d-dimensional
        index.
    d : int
        Dimension of index.

    Returns
    -------
    xstar_f : ndarray
        Formatted array. Strictly of shape (n,d) and F-ordered.

    Raises
    ------
    exceptions.ShapeError : if input array shape is not compatible with the
        model.
    """
    xstar = np.atleast_1d(xstar)
    if xstar.ndim == 1:
        if d == 1:
            xstar_f = xstar.reshape(-1, 1, order='F')
        elif xstar.shape[0] == d:
            xstar_f = xstar.reshape(1, -1, order='F')
        else:
            raise exceptions.ShapeError('xstar', 'n', 1, xstar.shape)
    elif xstar.ndim == 2:
        if xstar.shape[1] != d:
            raise exceptions.ShapeError('xstar', 'n', d, xstar.shape)
        xstar_f = utils._force_f_array(xstar)
    else:
        xstar_s = xstar.squeeze()
        if xstar_s.ndim == 2:
            return _format_input(xstar_s, d)
        raise exceptions.ShapeError('xstar', 'n', d, xstar.shape)
    return xstar_f


def defined_on_index(method: callable):
    """ Decorator for methods accepting arrays of points from the index.

    Applies _format_inputs to arrays passed to the wrapped method, ensuring
    arrays are correctly shaped and F-ordered before passing them to the
    decorated methods.

    Parameters
    ----------
    method : Emulator method
        Method bound to an Emulator instance accepting an (n,d) array xstar
        as its first argument.

    Returns
    -------
    wrapped_method : Emulator method
        Original method wrapped with defined_on_index, such that inputs are
        always correctly shaped.
    """
    @wraps(method)
    def wrapper(inst, xstar, *tail, **kwargs):
        xstar_f = _format_input(xstar, inst.d)
        return method(inst, xstar_f, *tail, **kwargs)
    return wrapper


# TODO: make kernel naming consistent with internal (use core.models.parse)
def _assign_internal(kernel_type):
    _AVAILABLE = {
        'eq': gp_internal.GpEqKernel,
        'm52': gp_internal.GpM52Kernel,
        'm32': gp_internal.GpM32Kernel,
        'rq': gp_internal.GpRqKernel
    }
    return _AVAILABLE[kernel_type]


class Emulator:

    """ Gaussian Process Emulator.

    User-facing interface to a compiled GP emulator.

    Attributes
    ----------
    x : ndarray
        (n,d) array of n sample points in d-dimensional space
    y : ndarray
        (n,) array of n sample responses, corresponding to the rows of x
    n : int
        Number of observations
    d : int
        Dimension of observations
    hyperparameters : dict
        Dictionary of model hyperparameters
    kernel_type : {'eq', 'm52', 'm32', 'rq'}, optional
        String designating the covariance kernel type:
        * 'eq' Exponentiated quadratic kernel
        * 'm52' Matern kernel with nu=5/2
        * 'm32' Matern kernel with nu=3/2
        * 'rq' Rational quadratic kernel
    mean_function_type : {'zero', None}, optional
        String designating the mean function type. Default = 'zero'
    info : dict
        Dictionary of diagnostic fit information
    m : int
        Number of hyperparameter samples

    Methods
    -------
    __call__(xstar)
        Posterior (predictive) expectation, integrated over hyperparameters
    expectation(xstar)
        Posterior (predictive) expectation (verbose wrapper for __call__),
        integrated over hyperparameters
    marginals(xstar)
        Posterior (predictive) marginal distributions at xstar for all m
        hyperparameters
    posterior(xstar)
        Posterior (predictive) joint distribution at xstar for all m
        hyperparameters
    loo_cv()
        Analytical leave-one-out cross validation scores for the n training
        sample points.
    ei(xstar)
        (Negative) expected improvement acquisition function, integrated over
        hyperparameters
    px(xstar)
        (Negative) posterior predictive variance ("pure exploration")
        acquisition function, integrated over hyperparameters
    ucb(xstar, beta)
        (Negative) upper confidence bound acquisition function, integrated over
        hyperparameters. This is a minimiser, so is strictly a "lower"
        confidence bound acquisition function.
    entropy(xstar)
        Differential entropy of predictive distribution at xstar, integrated
        over hyperparamters.
    optimisation.optimise(mode='min', x0=None, grid=None, grid_size=None,
        grid_method='lhs', grid_options=None, seed=None)
        Numerical optimisation (min or max) of posterior expectation.
    next_ei(x0=None, grid=None, grid_size=None, grid_method='lhs',
        grid_options=None, seed=None)
        Numerical optimisation of expected improvement.
    next_px(x0=None, grid=None, grid_size=None, grid_method='lhs',
        grid_options=None, seed=None)
        Numerical optimisation of posterior predictive variance.
    next_ucb(beta, x0=None, grid=None, grid_size=None, grid_method='lhs',
        grid_options=None, seed=None)
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
    _x : ndarray
        (n,d) array of n sample points in d dimensional space
    _y : ndarray
        (n,) array of n sample responses, corresponding to the rows in _x
    _gp :
        (Internal) c++ GP object.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            hyperparameters: typing.Dict[str, np.ndarray],
            info: typing.Optional[dict] = None,
            kernel_type: typing.Optional[str] = 'eq',
            mean_function_type: typing.Optional[str] = 'zero',
            jitter: typing.Optional[float] = 1e-10,
    ):
        """ Gaussian Process emulator.

        Parameters
        ----------
        x : ndarray
            (n, d) array of sample points.
        y : ndarray
            (n,) array of function responses.
        hyperparameters : dict
            Dictionary containing kernel hyperparameters.
        info : dict, optional
            Dictionary containing model fit information.
        kernel_type : {'eq', 'm52', 'm32', 'rq'}, optional
            String designating the kernel type. Default = 'eq'
        mean_function_type : {'zero', None}, optional
            String designating the mean function type. Default = 'zero'
        jitter : float, optional
            Magnitude of stability jitter. This is a standard deviation: supply
            the square root if designating a variance. Default = 1e-10.
        """

        # pylint: disable=too-many-arguments

        self._x = utils._force_f_array(x)
        self._y = utils._force_f_array(y)

        self.kernel_type = kernel_type
        self.mean_type = mean_function_type
        self.hyperparameters = hyperparameters

        try:
            self._amp = hyperparameters['amp']
            self._ls = hyperparameters['ls']
            self._sigma0 = hyperparameters['xi']

        except KeyError as missing_key:
            raise exceptions.MissingParameterError(str(missing_key)) from None

        # stability jitter
        self._delta = jitter

        # TODO fix this
        internal = _assign_internal(self.kernel_type)
        self._gp = internal(
            self._x,
            self._y,
            self._amp,
            self._ls,
            self._sigma0,
            self._delta,
        )

        self.n, self.d = x.shape
        self.m = self._amp.shape[0]

        # guess the fit method from m if not explicitly provided
        if info is None:
            info = {}
            if self.m > 1:
                info['method'] = 'hmc'
            else:
                info['method'] = 'mle'
        self.info = info

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @defined_on_index
    def __call__(self, xstar: np.ndarray):
        """ Posterior expectation.

        The posterior expectation of the emulator, integrated over the model
        hyperparameters.

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points at which to compute the posterior expectation

        Returns
        -------
        expectation : ndarray
            (n,) array corresponding to the posterior expectation at xstar

        Notes
        -----
        Identical to self.expectation(xstar)
        """
        return self._gp.E(xstar)

    def expectation(self, xstar: np.ndarray):
        """Posterior expectation

        The posterior expectation of the emulator, integrated over the model
        hyperparameters.

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points at which to compute the posterior expectation

        Returns
        -------
        expectation : ndarray
            (n,) array corresponding to the posterior expectation at xstar

        Notes
        -----
        Just an explicit wrapper for self.__call__
        """
        return self.__call__(xstar)

    @defined_on_index
    def marginals(self, xstar: np.ndarray):
        """"Predictive marginals

        Marginal predictive distributions corresponding to each hyperparameter
        sample at the points described by xstar.

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points at which to compute the predictive marginals

        Returns
        -------
        means : ndarray
            (n,m) array of means, corresponding to the mean at each of the n
            points in xstar for each of the m hyperparameter samples
        variances : ndarray
            (n,m) array of variances, corresponding to the variance at each of
            the n points in xstar for each of the m hyperparameter samples
        """
        return self._gp.marginals(xstar)

    @defined_on_index
    def posterior(self, xstar: np.ndarray):
        """Predictive distribution

        The joint predictive distribution of the model corresponding to each
        hyperparameter sample at the points described by xstar.

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points at which to compute the joint predictive
            distribution

        Returns
        -------
        means : ndarray
            (n,m) array of means, corresponding to the mean at each of the n
            points in xstar for each of the m hyperparameter samples
        covariance_matrices : ndarray
            (n,n,m) array consisting of m covariances matrices of size (n,n),
            describing the joint distribution over the n points in xstar for
            each of the m hyperparameter samples
        """
        return self._gp.posterior(xstar)

    def loo_cv(self):
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
    def ei(self, xstar: np.ndarray):
        """Expected improvement acquisition function

        The (negative) Expected Improvement (EI) acquisition function of
        Mockus [1]_, integrated over the model hyperparameters.

        This implementation is negative both in the sense that it seeks an
        improvement in the *minimum* value of the target function *and* that
        the acquisition function itself is to be minimised.

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points at which the expected improvement should be
            evaluated

        Returns
        -------
        expected_improvement : ndarray
            (n,) array containing the expected improvement at each of the n
            points in xstar, integrated over the hyperparameter samples

        See Also
        --------
        next_ei

        References
        ----------
        [1] Mockus, J., Tiesis, V. and Zilinskas, A., 1978. The application of
        bayesian methods for seeking the extremum. vol. 2.

        """
        return self._gp.ei(xstar)

    @defined_on_index
    def px(self, xstar: np.ndarray):
        """Pure exploration acquisition function

        The Pure eXploration (PX) acquisition function is equivalent to
        the (negative) predictive marginal variance integrated over the model
        hyperparameters.

        This implementation of PX is negative, and the point of maximum
        posterior marginal variance lies at the point in the index which
        minimises the PX function.

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points at which the negative predictive variance
            should be evaluated

        Returns
        -------
        px : ndarray
            (n,) array containing the negative predictive variance at each of
            the n points in xstar, integrated over the hyperparameter samples

        See Also
        --------
        next_px
        """
        return self._gp.px(xstar)

    @defined_on_index
    def ucb(self, xstar: np.ndarray, beta: float):
        """Upper confidence bound acquisition function

        The Upper Confidence Bound (UCB) acquisition function of
        Srinivas et. al. [1]_, integrated over the model hyperparameters.

        This implementation is negative both in the sense that it seeks an
        improvement in the *minimum* value of the target function *and* that
        the acquisition function itself is to be **minimised**.

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points at which the upper confidence bound function
            should be evaluated
        beta : float
             Parameter beta for the upper confidence bound acquisition function

        Returns
        -------
        ucb : ndarray
            (n,) array containing the upper confidence bound function at each
            of the n points in xstar, integrated over the hyperparameter
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
        return self._gp.ucb(xstar, beta)

    @defined_on_index
    def entropy(self, xstar: np.ndarray):
        """Differential entropy

        Compute the differential entropy of the posterior (joint) distribution
        at the points described by xstar.

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points at which the differential entropy of the
            posterior distribution should be evaluated

        Returns
        -------
        H : float
            Differential entropy
        """
        return self._gp.entropy(xstar)

    def optimise(
            self,
            mode: str = 'min',
            x0: typing.Optional[np.ndarray] = None,
            grid: typing.Optional[np.ndarray] = None,
            grid_size: typing.Optional[int] = None,
            grid_method: str = 'lhs',
            grid_options: typing.Optional[dict] = None,
            seed: typing.Optional[int] = None,
    ):
        """ Global Min/Max of the posterior expectation.

        Use numerical optimisation to estimate the global minimum or maximum of
        the expectation of the predictive distribution.

        Parameters
        ----------
        mode : {'min', 'max'}, optional
            If mode is 'min', attempt to find the global minimum of the
            predictive distribution. If mode is 'max', attempt to find the
            global maximum. Default = 'min'.
        x0 : {None, ndarray}, optional
            (d,) array representing the initialisation point for the
            optimisation routine. Since a gradient based optimiser is used,
            a good starting point helps to avoid convergence to a local
            optimum. If None, a preliminary grid search will be performed
            to determine a suitable initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify x0. If None, a grid will be generated. Default
            = None.
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
        """
        _mode = mode.lower()
        if _mode == 'min':
            obj = self._gp.E
            obj_jac = self._gp.E_jac
        elif _mode == 'max':
            # make objective and jacobian negative if searching for the max
            def obj(x):
                return -self._gp.E(x)

            def make_both_negative(f, jac):
                return (-f, -jac)

            def obj_jac(x):
                return make_both_negative(*self._gp.E_jac(x))

        else:
            raise ValueError("Mode must be either 'min' or 'max'.")
        opts = {
            'x0': x0,
            'grid': grid,
            'grid_size': grid_size,
            'grid_method': grid_method,
            'grid_options': grid_options,
            'seed': seed,
        }
        result = optimisation.optimise(obj, obj_jac, self.d, **opts)

        if _mode == 'max':
            result['xprime'] = -result['xprime']

        return result

    def next_ei(
            self,
            x0: typing.Optional[np.ndarray] = None,
            grid: typing.Optional[np.ndarray] = None,
            grid_size: typing.Optional[int] = None,
            grid_method: str = 'lhs',
            grid_options: typing.Optional[dict] = None,
            seed: typing.Optional[int] = None,
    ):
        """ Get next point using expected improvement.

        Use numerical optimisation to estimate the global minimum of the
        (negative) Expected Improvement (EI) acquisition function [1]_, hence
        determining the point in the index with the highest EI.

        Parameters
        ----------
        x0 : {None, ndarray}, optional
            (d,) array representing the initialisation point for the
            optimisation routine. Since a gradient based optimisation.optimiser
            is used, a good starting point helps to avoid convergence to a
            local optimum. If None, a preliminary grid search will be performed
            to determine a suitable initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify x0. If None, a grid will be generated. Default
            = None.
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

        See Also
        --------
        ei

        References
        ----------
        [1] Mockus, J., Tiesis, V. and Zilinskas, A., 1978. The application of
        bayesian methods for seeking the extremum. vol. 2.
        """

        f = self._gp.ei
        f_jac = self._gp.ei_jac
        opts = {
            'x0': x0,
            'grid': grid,
            'grid_size': grid_size,
            'grid_method': grid_method,
            'grid_options': grid_options,
            'seed': seed,
        }
        return optimisation.optimise(f, f_jac, self.d, **opts)

    def next_px(
            self,
            x0: typing.Optional[np.ndarray] = None,
            grid: typing.Optional[np.ndarray] = None,
            grid_size: typing.Optional[int] = None,
            grid_method: str = 'lhs',
            grid_options: typing.Optional[dict] = None,
            seed: typing.Optional[int] = None,
    ):
        """ Get next point using the pure exploration acquisition function.

        Use numerical optimisation to estimate the global minimum of the
        (negative) predictive variance (Pure eXploration; PX) acquisition
        function, hence determining the point in the index with the highest
        PX.

        Parameters
        ----------
        x0 : {None, ndarray}, optional
            (d,) array representing the initialisation point for the
            optimisation routine. Since a gradient based optimiser is used, a
            good starting point helps to avoid convergence to a local optimum.
            If None, a preliminary grid search will be performed to determine a
            suitable initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify x0. If None, a grid will be generated. Default
            = None.
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

        See Also
        --------
        px
        """

        f = self._gp.px
        f_jac = self._gp.px_jac
        opts = {
            'x0': x0,
            'grid': grid,
            'grid_size': grid_size,
            'grid_method': grid_method,
            'grid_options': grid_options,
            'seed': seed,
        }
        return optimisation.optimise(f, f_jac, self.d, **opts)

    def next_ucb(
            self,
            beta: float,
            x0: typing.Optional[np.ndarray] = None,
            grid: typing.Optional[np.ndarray] = None,
            grid_size: typing.Optional[int] = None,
            grid_method: str = 'lhs',
            grid_options: typing.Optional[dict] = None,
            seed: typing.Optional[int] = None,
    ):
        """ Get next point using upper confidence bound

        Use numerical optimisation to estimate the global minimum of the
        (negative) Upper Confidence Bound (UCB) acquisition function [1]_,
        hence determining the point in the index with the highest UCB.

        Parameters
        ----------
        beta : float
            Parameter beta for the upper confidence bound acquisition function
        x0 : {None, ndarray}, optional
            (d,) array representing the initialisation point for the
            optimisation routine. Since a gradient based optimiser is used, a
            good starting point helps to avoid convergence to a local optimum.
            If None, a preliminary grid search will be performed to determine a
            suitable initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify x0. If None, a grid will be generated. Default
            = None.
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

        See Also
        --------
        ucb

        References
        ----------
        [1] Srinivas, N., Krause, A., Kakade, S.M. and Seeger, M., 2009.
        Gaussian process optimization in the bandit setting: No regret and
        experimental design. arXiv preprint arXiv:0912.3995.
        """

        # closures define objective function and jacobian with desired beta
        def obj(x):
            return self._gp.ucb(x, float(beta))

        def obj_jac(x):
            return self._gp.ucb_jac(x, float(beta))

        opts = {
            'x0': x0,
            'grid': grid,
            'grid_size': grid_size,
            'grid_method': grid_method,
            'grid_options': grid_options,
            'seed': seed,
        }
        return optimisation.optimise(obj, obj_jac, self.d, **opts)

    def sobol1(
            self,
            n: typing.Optional[int] = 1000,
            method: typing.Optional[str] = 'sobol',
            seed: typing.Optional[int] = None,
    ):
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

        References
        ----------
        [1] Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto,
        M. and Tarantola, S., 2010. Variance based sensitivity analysis of
        model output. Design and estimator for the total sensitivity index.
        Computer Physics Communications, 181(2), pp.259-270.
        """

        X = sampling.sample_hypercube(n, 2*self.d, method=method, seed=seed)
        A = X[:, :self.d]
        B = X[:, self.d:]
        f_A = self.expectation(A)
        f_B = self.expectation(B)
        V_i = np.empty(self.d)
        f_all = [f_A, f_B]
        for i in range(self.d):
            AB = A.copy()
            AB[:, i] = B[:, i]
            f_i = self.expectation(AB)
            V_i[i] = np.mean(f_B * (f_i - f_A))  # local variance
            f_all.append(f_i)
        V_T = np.var(np.array(f_all).ravel())
        return V_i / V_T

    def plot_parameter(self, param_name: str):
        """ Trace plot of hyperparameter with param_name"""
        if self.info['method'] == 'hmc':
            visualisation.plot_parameter(
                self.hyperparameters,
                param_name,
                self.info
            )
        else:
            raise RuntimeError('Method only valid for models fit using HMC.')

    def plot_divergences(self):
        """ Parallel co-ordinates plot of sampler behaviour

        Highlights any divergent transitions.
        """
        if self.info['method'] == 'hmc':
            visualisation.plot_divergences(
                self.hyperparameters,
                self.info
            )
        else:
            raise RuntimeError('Method only valid for models fit using HMC.')
