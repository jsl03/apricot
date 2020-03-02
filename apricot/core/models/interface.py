from apricot.core.models.build import (
    make_kernel,
    make_mean,
    make_noise,
)

from apricot.core.models.parse import (
    parse_kernel,
    parse_mean,
    parse_noise,
    parse_warping,
)

from apricot.core.models.initialisation import (
    make_pystan_dict,
    get_init,
)

from apricot.core.models.mle import run_mle
from apricot.core.models.hmc import run_hmc
from apricot.core.models.model_cache import load
from apricot.core.utils import flatten

class Interface(object):

    """ apricot interface to a pyStan GP model."""

    def __init__(self, kernel, mean_function=None, noise=None, warping=None):

        """ apricot interface to a pyStan GP model

        Parameters
        ----------
        kernel : str

        mean_function : str, optional

        noise : str, optional

        warping : str, optional

        Returns
        -------
        pyStan_interface : apricot.core.Interface instance

        """

        self.kernel_type = parse_kernel(kernel)
        self.mean_function_type = parse_mean(mean_function)
        self.noise_type = parse_noise(noise)
        self.warping = parse_warping(warping)
        warp = bool(warping)

        kernel_part = make_kernel(self.kernel_type, warp)
        mean_part = make_mean(self.mean_function_type)
        noise_part = make_noise(self.noise_type)

        self.pystan_model = load(
            kernel_part,
            mean_part,
            noise_part,
            warp,
        )

        # required arguments for kernel, mean function and noise function
        self.theta = [a[0] for a in kernel_part.args]
        self.beta = [a[0] for a in mean_part.args]
        self.xi = [a[0] for a in noise_part.args]

        # dimensions of the required arguments
        self._theta_dims = [a[1] for a in kernel_part.args]
        self._beta_dims = [a[1] for a in mean_part.args]
        self._xi_dims = [a[1] for a in noise_part.args]

        # parameters to sample
        self._pars_to_sample = flatten((
            kernel_part.to_sample,
            mean_part.to_sample,
            noise_part.to_sample,
            ['lp__']
        ))

        # required data to run the sampler
        self._data_required = flatten((
            kernel_part.data_priors,
            mean_part.data_priors,
            noise_part.data_priors
        ))

    def make_pystan_dict(self, x, y, jitter=1e-10, fit_options=None, seed=None):
        """ Construct the pystan 'data' dictionary

        Parameters
        ----------
        x : ndarray
            (n,d) array with each row representing a sample point in d-dimensional
            space.
        y : ndarray
            (n,) array of responses corresponding to each row of x.
        jitter : float, optional
            Stability jitter. Default = 1e-10.
        seed : {int32, None}
            Random seed.

        Returns
        -------
        stan_dict : dict
            pystan data dictionary. The pystan model is then executed via
            <instance>.sampling(data=stan_dict) for hmc or
            <instance>.optimizing(data = stan_dict) for mle.

        Notes
        -----
        This is a wrapper for models.initialisation.make_pystan_dict
        """
        return make_pystan_dict(self, x, y, jitter, fit_options, seed=seed)

    def get_init(self, stan_dict, init):
        """ Create dictionary of initial values for pyStan

        Parameters
        ----------
        stan_dict : dict
            Data dictionary to be passed to pyStan, obtained either via
            <instance>.make_pystan_dict or supplied manually.
        init : {dict, str}
            * if init is a dict, it is assumed to contain an initial value for each
            of the parameters to be sampled by the pyStan model.
            * if init is None, the init method defaults to 'stable'.
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
        This is a wrapper for models.initialisation.get_init 
        """
        return get_init(self, stan_dict, init)

    def hmc(self, *args, **kwargs):
        """ Sample model hyperparameters using Hamiltonian Monte-Carlo

        Parameters
        ----------
        x : ndarray
            (n,d) array with each row representing a sample point in d-dimensional
            space.
        y : ndarray
            (n,) array of responses corresponding to each row of x.
        jitter : float, optional
            Stability jitter. Default = 1e-10.
        fit_options : dict, optional
        samples : int, optional
            Number of samples to draw from the posterior (accounting for the number
            of chains, warmup and thinning). Default = 2000.
        thin : int
            If > 1, keep only every thin samples. Default = 1.
        chains : int
            The number of independent chains to draw samples from. Default = 4.
        adapt_delta : float < 1
            Adapt_delta control parameter to the sampler. Default = 0.8.
        max_treedepth : int
            Maximum sample tree depth control parameter. Default = 10.
        seed : {int32, None}
            Random seed.
        permute : bool, optional
            If True, permute the samples.
        init : {dict, str}
            * if init is a dict, it is assumed to contain an initial value for each
            of the parameters to be sampled by the pyStan model.
            * if init is None, the init method defaults to 'stable'.
            * if init is a string, it is matched to one of the following:
                - 'stable' : initialise from data. Lengthscales are initialised to
                the standard deviation of the respective column of x.
                - {'0' , 'zero} : initialise all parameters as 0.
                - 'random' : initialise all parameters randomly on their
                support.
            Default = 'stable'.

        Returns
        -------
        Parameters : dict
            Dictionary containing the sampled hyperparameters
        info : dict
            Diagnostic information.

        Notes
        -----
        This is a wrapper for models.hmc.run_hmc
        """
        parameters, info = run_hmc(self, *args, **kwargs)
        return parameters, info

    def mle(self, *args, **kwargs):
        """ Identify model parameters using log-likelihood optimisation

        Parameters
        ----------
        x : ndarray
            (n,d) array with each row representing a sample point in d-dimensional
            space.
        y : ndarray
            (n,) array of responses corresponding to each row of x.
        jitter : float, optional
             Magnitude of stability jitter. Default = 1e-10.
        fit_options :
             Optional extra parameters to the GP prior distribution.
        init_method : {dict, str}
            * if init is a dict, it is assumed to contain an initial value for each
            of the parameters to be sampled by the pyStan model.
            * if init is None, the init method defaults to 'stable'.
            * if init is a string, it is matched to one of the following:
                - 'stable' : initialise from data. Lengthscales are initialised to
                the standard deviation of the respective column of x.
                - {'0' , 'zero} : initialise all parameters as 0.
                - 'random' : initialise all parameters randomly on their
                support.
            Defaults to 'random' if restarts > 1, and stable otherwise.
        algorithm : str, optional
             String specifying which of Stan's gradient based optimisation
             algorithms to use. Default = 'Newton'.
        restarts : int, optional
             The number of restarts to use. The optimisation will be repeated
             this many times and the hyperparameters with the highest
             log-likelihood will be returned. restarts > 1 is not compatible with
             'stable' or 0 initialisations. Default=10.
        max_iter : int, optional
             Maximum allowable number of iterations for the chosen optimisation
             algorithm. Default = 250.

        Returns
        -------
        Parameters : dict
            Dictionary containing the optimised model hyperparameters.
        info : dict
            Diagnostic information.

        Notes
        -----
        This is a wrapper for models.mle.run_mle
        """
        parameters, info = run_mle(self, *args, **kwargs)
        return parameters, info
