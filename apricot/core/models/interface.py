"""
Code for the primary interface between the user, apricot models, and pyStan.

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""


from typing import Optional, Union, Dict, Any, Tuple
import numpy as np  # type: ignore
from apricot.core import utils
from apricot.core.models import build
from apricot.core.models import parse
from apricot.core.models import initialisation
from apricot.core.models import map as maxap  # avoid masking map
from apricot.core.models import hmc
from apricot.core.models import model_cache
from apricot.core.models import type_aliases as ta


class Interface:
    # pylint: disable=too-many-instance-attributes
    """ apricot interface to a pyStan GP model."""

    def __init__(
            self,
            kernel: str,
            mean_function: Optional[ta.MeanFunction] = None,
            noise: Optional[ta.NoiseModel] = None,
            warping: Optional[str] = None,
    ) -> None:
        """ apricot interface to a pyStan GP model

        Parameters
        ----------
        kernel: str
            The desired covariance kernel. One of:
            * 'eq': Exponentiated quadratic.
            * 'eq_flat': Exponentiated quadratic with flat (i.e., uniform)
                hyperparameter priors. Used for debug purposes and model
                tests.
            * 'm52': Matern kernel with nu = 5/2.
            * 'm32': Matern kernel with nu = 3/2.
            * 'rq': Rational quadratic kernel.
        mean_function: str, optional
            The desired mean function. One of:
            * 0: Zero mean.
            * 'zero': see above.
            * 'linear: linear mean, i.e. mu = beta * x
        noise: str, optional
            The desired noise model. If noise is a floating point number,
            additive Gaussian (white) noise with standard deviation equivalent
            to noise will be added to the leading diagonal of the sample
            covariance matrix. If noise = 'infer', the standard deviation of
            additive Gaussian noise will be inferred as a model hyperparameter.
        warping: {None, False, str}, optional
            If None or False, no input warping will be applied. If warping =
            'linear' or warping = 'sigmoid', input warping using the Beta
            distribution CDF will be applied as described in [1]_. Prior
            distributions on the Beta distribution's parameters are used to
            encode prior belief that the warping is approximately linear or
            sigmoidal, respectively.

        Returns
        -------
        pyStan_interface: apricot.core.Interface instance

        References
        ----------
        [1] Snoek, Jasper, et al. "Input warping for Bayesian optimization of
        non-stationary functions." International Conference on Machine
        Learning. 2014.
        """

        self.kernel_type = parse.parse_kernel(kernel)
        self.mean_function_type = parse.parse_mean(mean_function)
        self.noise_type = parse.parse_noise(noise)
        self.warping = parse.parse_warping(warping)

        kernel_part = build.make_kernel(self.kernel_type, self.warping)
        mean_part = build.make_mean(self.mean_function_type)
        noise_part = build.make_noise(self.noise_type[0])

        warp = bool(self.warping)

        self.pystan_model = model_cache.load(
            kernel_part,
            mean_part,
            noise_part,
            warp,
        )

        # required arguments for kernel, mean function and noise function
        self.theta = [a[0] for a in kernel_part.args]
        self.beta = [a[0] for a in mean_part.args]
        self.sigma = [a[0] for a in noise_part.args]

        # dimensions of the required arguments
        self._theta_dims = [a[1] for a in kernel_part.args]
        self._beta_dims = [a[1] for a in mean_part.args]
        self._sigma_dims = [a[1] for a in noise_part.args]

        # the parameters to sample
        self.pars_to_sample = utils.flatten((
            kernel_part.to_sample,
            mean_part.to_sample,
            noise_part.to_sample,
            ['lp__']
        ))
        # the data required to run the sampler
        self._data_required = utils.flatten((
            kernel_part.data_priors,
            mean_part.data_priors,
            noise_part.data_priors
        ))

    def make_pystan_dict(  # pylint: disable=too-many-arguments
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            jitter: float = 1e-10,
            ls_options: Optional[ta.LsPriorOptions] = None,
            seed: Optional[int] = None
    ) -> ta.PyStanData:
        """ Construct the pystan 'data' dictionary

        Parameters
        ----------
        x_data : ndarray
            (n,d) array with each row representing a sample point in
            d-dimensional space.
        y_data : ndarray
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
            <instance>.optimizing(data = stan_dict) for map.

        Notes
        -----
        This is a wrapper for models.initialisation.make_pystan_dict
        """
        return initialisation.make_pystan_dict(
            self,
            x_data,
            y_data,
            jitter,
            ls_options,
            seed=seed
        )

    def get_init(
            self,
            init_method: ta.InitTypes,
            stan_dict: ta.PyStanData
    ) -> ta.InitTypes:
        """ Create dictionary of initial values for pyStan

        Parameters
        ----------
        stan_dict : dict
            Data dictionary to be passed to pyStan, obtained either via
            <instance>.make_pystan_dict or supplied manually.
        init : {dict, str}
            * if init is a dict, it is assumed to contain an initial value for
                each of the parameters to be sampled by the pyStan model.
            * if init is None, the init method defaults to 'stable'.
            * if init is a string, it is matched to one of the following:
                - 'stable' : initialise from data. Lengthscales are initialised
                    to the standard deviation of the respective column of x.
                - {'0' , 'zero} : initialise all parameters as 0.
                - 'random' : initialise all parameters randomly on their
                    support.

        Returns
        -------
        init : {dict, str, int}
            Initialisation values for the parameters for the desired
            pyStan model.

        Notes
        -----
        This is a wrapper for models.initialisation.get_init
        """
        return initialisation.get_init(self, init_method, stan_dict)

    def hmc(  # pylint: disable=too-many-arguments, too-many-locals
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            jitter: float = 1e-10,
            ls_options: Optional[ta.LsPriorOptions] = None,
            samples: int = 2000,
            thin: int = 1,
            chains: int = 4,
            adapt_delta: float = 0.8,
            max_treedepth: int = 10,
            seed: Optional[int] = None,
            permute: bool = True,
            init_method: ta.InitTypes = 'stable',
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Sample model hyperparameters using Hamiltonian Monte-Carlo

        Parameters
        ----------
        x_data : ndarray
            (n,d) array with each row representing a sample point in
            d-dimensional space.
        y_data : ndarray
            (n,) array of responses corresponding to each row of x.
        jitter : float, optional
            Stability jitter. Default = 1e-10.
        fit_options : dict, optional
        samples : int, optional
            Number of samples to draw from the posterior (accounting for the
            number of chains, warmup and thinning). Default = 2000.
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
            * if init is a dict, it is assumed to contain an initial value for
                each of the parameters to be sampled by the pyStan model.
            * if init is a string, it is matched to one of the following:
                - 'stable' : initialise from data. Lengthscales are initialised
                    to the standard deviation of the respective column of x.
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
        parameters, info = hmc.run_hmc(
            self,
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
            init_method=init_method,
        )
        return parameters, info

    def map(  # pylint: disable=too-many-arguments, too-many-locals
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            jitter: float = 1e-10,
            ls_options: Optional[ta.LsPriorOptions] = None,
            init_method: ta.InitTypes = 'stable',
            algorithm: str = 'Newton',
            restarts: int = 10,
            max_iter: int = 250,
            seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """ Identify model parameters using log-likelihood optimisation

        Parameters
        ----------
        x_data : ndarray
            (n,d) array with each row representing a sample point in
            d-dimensional space.
        y_data : ndarray
            (n,) array of responses corresponding to each row of x.
        jitter : float, optional
             Magnitude of stability jitter. Default = 1e-10.
        fit_options :
             Optional extra parameters to the GP prior distribution.
        init_method : {'stable', 'random', 0, dict}
            String determining the initialisation method. Note that if restarts
            > 1, only the first optimisation is initialised according to
            init_method, and the rest will be initialised using
            init_method = 'random':
            * 'stable' : "stable" initialise parameters from "stable" guesses.
            * 'zero' : initialise all parameters from zero.
            * 'random' : initialise all parameters randomly on their support.
            * dict : A custom initialisation value for each of the model's
                parameters.
        Default = 'random'.
        algorithm : str, optional
             String specifying which of Stan's gradient based optimisation
             algorithms to use. Default = 'Newton'.
        restarts : int, optional
             The number of restarts to use. The optimisation will be repeated
             this many times and the hyperparameters with the highest
             log-likelihood will be returned. restarts > 1 is not compatible
             with 'stable' or 0 initialisations. Default=10.
        max_iter : int, optional
             Maximum allowable number of iterations for the chosen optimisation
             algorithm. Default = 250.
        seed : {int32, None}
            Random seed.

        Returns
        -------
        Parameters : dict
            Dictionary containing the optimised model hyperparameters.
        info : dict
            Diagnostic information.

        Notes
        -----
        This is a wrapper for models.map.run_map
        """
        parameters, info = maxap.run_map(
            self,
            x_data,
            y_data,
            jitter=jitter,
            ls_options=ls_options,
            init_method=init_method,
            algorithm=algorithm,
            restarts=restarts,
            max_iter=max_iter,
            seed=seed,
        )
        return parameters, info
