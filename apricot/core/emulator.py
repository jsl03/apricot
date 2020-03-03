import numpy as np
from functools import wraps

from apricot.core.gp import Gp
from apricot.core.optimisation import optimise
from apricot.core.utils import _force_f_array, _atleast2d_fview
from apricot.core.exceptions import ShapeError, MissingParameter

from apricot.core.visualisation import plot_parameter as _plot_parameter
from apricot.core.visualisation import plot_divergences as _plot_divergences

def _format_input(xstar, d):
    """ Format inputs

    Ensures arrays are correctly shaped and F-ordered before passing to the
    internal GP methods to prevent repeated copying / c++ runtime errors.

    Parameters
    ----------
    xstar : ndarray
        Unformatted array. Raw (n,d) array of n points from the d-dimensional
        index.
    d : int
        Dimension of Emulator index.

    Returns
    -------
    xstar_f : ndarray
        Formatted array. Strictly of shape (n,d) and F-ordered.

    Raises
    ------
    ShapeError : if input array shape is not compatible with the model.
    """
    xstar = np.atleast_1d(xstar)
    if xstar.ndim==1:
        if d==1:
            xstar_f = xstar.reshape(-1, 1, order='F')
        elif xstar.shape[0] == d:
            xstar_f = xstar.reshape(1, -1, order='F')
        else:
            raise ShapeError('xstar', 'n', 1, xstar.shape)
    elif xstar.ndim == 2:
        if xstar.shape[1] != d:
            raise ShapeError('xstar', 'n', d, xstar.shape)
        xstar_f = _force_f_array(xstar)
    else:
        xstar_s = xstar.squeeze()
        if xstar_s.ndim == 2:
            return _format_input(xstar_s, d)
        else:
            raise ShapeError('xstar', 'n', d, xstar.shape)
    return xstar_f

def _defined_on_index(method):
    """Decorator for methods accepting arrays of points from the index.

    Applies _format_inputs to arrays passed to the wrapped method, ensuring
    arrays are correctly shaped and F-ordered before passing them to the
    internal GP.
    """
    @wraps(method)
    def wrapper(inst, xstar, *tail, **kwargs):
        xstar_f = _format_input(xstar, inst.d)
        return method(inst, xstar_f, *tail, **kwargs)
    return wrapper

class Emulator:
    """ Gaussian Process Emulator

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
    optimise(mode='min', x0=None, grid=None, grid_size=None, grid_method='lhs',
    grid_options=None)
        Numerical optimisation (min or max) of posterior expectation.
    next_ei(x0=None, grid=None, grid_size=None, grid_method='lhs',
    grid_options=None)
        Numerical optimisation of expected improvement.
    next_px(x0=None, grid=None, grid_size=None, grid_method='lhs',
    grid_options=None)
        Numerical optimisation of posterior predictive variance.
    next_ucb(beta, x0=None, grid=None, grid_size=None, grid_method='lhs',
    grid_options=None)
        Numerical optimisation of upper confidence bound.

    Private Attributes
    ------------------
    _gp :
        (Internal) c++ GP object.
    """

    def __init__(self, x, y, hyperparameters, info, jitter=1e-10, kernel=None, mean_function=None):
        """ GP emulator

        Parameters
        ----------
        x : ndarray
            (n, d) array of sample points.
        y : ndarray
            (n,) array of function responses.
        hyperparameters : dict
            Dictionary containing kernel hyperparameters.
        info : dict
            Dictionary containing model fit information.
        jitter : float, optional
            Magnitude of stability jitter. This is a standard deviation: supply
            the square root if designating a variance. Default = 1e-10.
        """

        self._x = _force_f_array(x)
        self._y = _force_f_array(y)

        self.hyperparameters = hyperparameters
        self.fit_info = info
        self.fit_method = info['method']

        # ---------------------------------------------------------------------
        # Temporary fix for more flexible parametrisation. Note internal GP
        # wants squared hyperparameters. Once more kernels are added we need
        # to check what the internal GP expects.
        try:
            self._amp_sq = hyperparameters['amp']**2
            self._ls_sq = hyperparameters['ls']**2
            self._sigma0 = hyperparameters['xi']**2
            
        except KeyError as ke:
            raise MissingParameter(str(ke)) from None
        # ---------------------------------------------------------------------

        # stability jitter
        self._delta = jitter**2

        self._gp = Gp(
            self._x,
            self._y,
            self._amp_sq,
            self._ls_sq,
            self._sigma0,
            self._delta,
        )

        self.n, self.d = x.shape
        self.m = self._amp_sq.shape[0]

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @_defined_on_index
    def __call__(self, xstar):
        """Posterior Expectation

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

    def expectation(self, xstar):
        """Posterior Expectation

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

    @_defined_on_index
    def marginals(self, xstar):
        """"Predictive Marginals

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

    @_defined_on_index
    def posterior(self, xstar):
        """Predictive distribution

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points at which the joint predictive distribution
            is sought

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
        samples using the method of Sundararajan & Keerthi _[1].

        Returns
        -------
        cv_scores : ndarray

        References
        ----------
        [1] Sundararajan, S. and Keerthi, S.S., 2000.
        Predictive approaches for choosing hyperparameters in Gaussian processes.
        In Advances in neural information processing systems (pp. 631-637).
        """
        return self._gp.loo_cv()

    @_defined_on_index
    def ei(self, xstar):
        """Expected Improvement Acquisition Function

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

        """
        return self._gp.ei(xstar)

    @_defined_on_index
    def px(self, xstar):
        """Pure Exploration Acquisition Function

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

        Notes
        -----
        Equivalent to the expectation of the negative marginal variance
        """
        return self._gp.px(xstar)

    @_defined_on_index
    def ucb(self, xstar, beta):
        """Upper Confidence Bound Acquisition Function

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
            (n,) array containing the upper confidence bound function at each of
            the n points in xstar, integrated over the hyperparameter samples

        Notes
        -----
        Strictly speaking this is a "lower confidence bound" acquisition
        function, as the acquisition function seeks the minimum
        """
        return self._gp.ucb(xstar, beta)

    @_defined_on_index
    def entropy(self, xstar):
        """Differential Entropy

        Parameters
        ----------
        xstar : ndarray
            (n,d) array of points for which the differential entropy of the
            posterior should be evaluated

        Returns
        -------
        H : float
            Differential entropy
        """
        return self._gp.entropy(xstar)

    def optimise(self, mode='min', x0=None, grid=None, grid_size=None,
                 grid_method='lhs', grid_options=None, seed=None):
        """ Global Min/Max of the posterior expectation

        Use numerical optimisation to estimate the global minimum or maximum of
        the predictive expectation.

        Parameters
        ----------
        mode : {'min', 'max'}, optional
            If mode is 'min', attempt to find the global minimum of the
            predictive distribution. If mode is 'max', attempt to find the global
            maximum. Default = 'min'.
        x0 : {None, ndarray}, optional
            (d,) array representing the initialisation point for the optimisation
            routine. Since a gradient based optimiser is used, a good starting
            point helps to avoid convergence to a local optimum. If None, a
            preliminary grid search will be performed to determine a suitable
            initialisation point. Default = None.
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
            f = self._gp.E
            f_jac = self._gp.E_jac
        elif _mode == 'max':
            # need wrapper to make E negative in this case
            f = lambda x : -self._gp.E(x)
            # make both E and it's Jacobian negative
            make_both_negative = lambda f, jac : (-f, -jac)
            f_jac = lambda x : make_both_negative(*self._gp.E_jac(x))
        else:
            raise ValueError("Mode must be either 'min' or 'max'.")
        opts = {
            'x0' : x0,
            'grid' : grid,
            'grid_size' : grid_size,
            'grid_method' : grid_method,
            'grid_options' : grid_options,
            'seed' : seed,
        }
        result = optimise(f, f_jac, self.d, **opts)
        if _mode == 'max':
            result['xprime'] = -result['xprime']
        return result

    def next_ei(self, x0=None, grid=None, grid_size=None, grid_method='lhs',
                grid_options=None, seed=None):
        """ Get next point using expected improvement

        Use numerical optimisation to estimate the global minimum of the
        (negative) expected improvement (EI) acquisition function, hence
        determining the point in the index with the highest EI.

        Parameters
        ----------
        x0 : {None, ndarray}, optional
            (d,) array representing the initialisation point for the optimisation
            routine. Since a gradient based optimiser is used, a good starting
            point helps to avoid convergence to a local optimum. If None, a
            preliminary grid search will be performed to determine a suitable
            initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify x0. If None, a grid will be generated. Default
            = None.
        grid_size : {None, int}, optional
            Integer specifying the number of points to use in the preliminary
            grid search. If None, use 100*d. Default = None.
        grid_method : str, optional
            String specifying the experimental design strategy used to
            construct the grid of points used to conduct the grid search. If None,
            defaults to 'lhs'. Default = None.
        grid_options : dict, optional
            Dictionary of additional options to pass to grid_method. Default
            = None.
        seed : {None, int32}
            Random seed. Used for preliminary grid generation. Default = None.

        See Also
        --------
        ei
        """

        f = self._gp.ei
        f_jac = self._gp.ei_jac
        opts = {
            'x0' : x0,
            'grid' : grid,
            'grid_size' : grid_size,
            'grid_method' : grid_method,
            'grid_options' : grid_options,
            'seed' : seed,
        }
        return optimise(f, f_jac, self.d, **opts)

    def next_px(self, x0=None, grid=None, grid_size=None, grid_method='lhs',
                grid_options=None, seed=None):
        """ Get next point using the pure exploration acquisition function

        Use numerical optimisation to estimate the global minimum of the
        (negative) predictive variance (pure exploration; PX) acquisition
        function, hence determining the point in the index with the highest
        PX.

        Parameters
        ----------
        x0 : {None, ndarray}, optional
            (d,) array representing the initialisation point for the optimisation
            routine. Since a gradient based optimiser is used, a good starting
            point helps to avoid convergence to a local optimum. If None, a
            preliminary grid search will be performed to determine a suitable
            initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify x0. If None, a grid will be generated. Default
            = None.
        grid_size : {None, int}, optional
            Integer specifying the number of points to use in the preliminary
            grid search. If None, use 100*d. Default = None.
        grid_method : str, optional
            String specifying the experimental design strategy used to
            construct the grid of points used to conduct the grid search. If None,
            defaults to 'lhs'. Default = None.
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
            'x0' : x0,
            'grid' : grid,
            'grid_size' : grid_size,
            'grid_method' : grid_method,
            'grid_options' : grid_options,
            'seed' : seed,
        }
        return optimise(f, f_jac, self.d, **opts)

    def next_ucb(self, beta, x0=None, grid=None, grid_size=None, grid_method='lhs',
                 grid_options=None, seed=None):
        """ Get next point using the upper confidence bound acquisition function

        Use numerical optimisation to estimate the global minimum of the
        (negative) upper confidence bound (UCB) acquisition function,
        hence determining the point in the index with the highest UCB.

        Parameters
        ----------
        beta : float
            Parameter beta for the upper confidence bound acquisition function
        x0 : {None, ndarray}, optional
            (d,) array representing the initialisation point for the optimisation
            routine. Since a gradient based optimiser is used, a good starting
            point helps to avoid convergence to a local optimum. If None, a
            preliminary grid search will be performed to determine a suitable
            initialisation point. Default = None.
        grid : {None, ndarray}, optional
            (n,d) array of n points at which to perform the preliminary grid
            search to identify x0. If None, a grid will be generated. Default
            = None.
        grid_size : {None, int}, optional
            Integer specifying the number of points to use in the preliminary
            grid search. If None, use 100*d. Default = None.
        grid_method : str, optional
            String specifying the experimental design strategy used to
            construct the grid of points used to conduct the grid search. If None,
            defaults to 'lhs'. Default = None.
        grid_options : dict, optional
            Dictionary of additional options to pass to grid_method. Default
            = None.
        seed : {None, int32}
            Random seed. Used for preliminary grid generation. Default = None.

        See Also
        --------
        ucb
        """

        # partially apply beta to get objective function and derivatives
        f = lambda x : self._gp.ucb(x, float(beta))
        f_jac = lambda x :self._gp.ucb_jac(x, float(beta))
        opts = {
            'x0' : x0,
            'grid' : grid,
            'grid_size' : grid_size,
            'grid_method' : grid_method,
            'grid_options' : grid_options,
            'seed' : seed,
        }
        return optimise(f, f_jac, self.d, **opts)

    def sobol1(self, n=1000, method='sobol', seed=None):
        """ Calculate first order Sobol indices for the emulator.

        Approximates first order Sobol indices for the emulator using Monte-Carlo
        simulation as described in _[1].

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

        # generate the sample points and split into arrays A and B
        X = apricot.sample_hypercube(n, 2*self.d, method=method, seed=seed)
        A = X[:,:self.d]
        B = X[:,self.d:]

        # get the expectation of the emulator for points in A and B
        f_A = self.expectation(A)
        f_B = self.expectation(B)

        # initialise
        V_i = np.empty(self.d)

        # store all the function responses for estimating the global variance
        f_all = [f_A, f_B]

        # cycle over index dimensions
        for i in range(self.d):
            AB = A.copy()

            # column i of A is replaced with column i from B to form array AB
            AB[:,i] = B[:,i]

            # get the expectation for points in AB
            f_i = self.expectation(AB)

            # compute the local variance
            V_i[i] = np.mean(f_B * (f_i - f_A))

            # store result
            f_all.append(f_i)

        # estimate the total variance
        V_T = np.var(np.array(f_all).ravel())

        # divide local variances by total variance and we're done
        return V_i / V_T

    def plot_parameter(self, param_name):
        """ Traceplot of hyperparameter with param_name"""
        if self.fit_method == hmc:
            _plot_parameter(self.hyperparameters, param_name, self.fit_info)
        else:
            raise RuntimeError('method is only valid for models fit using HMC.')

    def plot_divergences(self):
        """ Parallel co-ordinates plot of sampler behaviour

        Highlights any divergent transitions.
        """
        if self.fit_method == hmc:
            _plot_divergences(self.hyperparameters, self.fit_info)
        else:
            raise RuntimeError('method is only valid for models fit using HMC.')
