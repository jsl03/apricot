# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
import typing
import numpy as np
from apricot.core import utils
from apricot.core.models import build
from apricot.core.models import parse
from apricot.core.models import initialisation
from apricot.core.models import map as maxap  # avoid masking map
from apricot.core.models import hmc
from apricot.core.models import model_cache


class Interface(object):

    """ apricot interface to a pyStan GP model."""

    def __init__(
            self,
            kernel: str,
            mean_function: typing.Optional[typing.Union[str, int]] = None,
            noise: typing.Optional[typing.Union[str, float]] = None,
            warping: typing.Optional[str] = None,
    ) -> None:
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

        self.kernel_type = parse.parse_kernel(kernel)
        self.mean_function_type = parse.parse_mean(mean_function)
        self.noise_type = parse.parse_noise(noise)
        self.warping = parse.parse_warping(warping)
        warp = bool(warping)

        kernel_part = build.make_kernel(self.kernel_type, warp)
        mean_part = build.make_mean(self.mean_function_type)
        noise_part = build.make_noise(self.noise_type)

        self.pystan_model = model_cache.load(
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
        # TODO: attributes probably redundant as of 01.01.2020
        self._theta_dims = [a[1] for a in kernel_part.args]
        self._beta_dims = [a[1] for a in mean_part.args]
        self._xi_dims = [a[1] for a in noise_part.args]

        # parameters to sample
        self._pars_to_sample = utils.flatten((
            kernel_part.to_sample,
            mean_part.to_sample,
            noise_part.to_sample,
            ['lp__']
        ))

        # required data to run the sampler
        self._data_required = utils.flatten((
            kernel_part.data_priors,
            mean_part.data_priors,
            noise_part.data_priors
        ))

    def make_pystan_dict(
            self,
            x: np.ndarray,
            y: np.ndarray,
            jitter: float = 1e-10,
            fit_options: typing.Optional[dict] = None,
            seed: typing.Optional[int] = None
    ) -> dict:
        """ Construct the pystan 'data' dictionary

        Parameters
        ----------
        x : ndarray
            (n,d) array with each row representing a sample point in
            d-dimensional space.
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
            <instance>.optimizing(data = stan_dict) for map.

        Notes
        -----
        This is a wrapper for models.initialisation.make_pystan_dict
        """
        return initialisation.make_pystan_dict(
            self,
            x,
            y,
            jitter,
            fit_options,
            seed=seed
        )

    def get_init(
            self,
            init_method: typing.Optional[typing.Union[str, dict]],
            stan_dict: dict
    ) -> typing.Union[dict, str, int]:
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

    def hmc(
            self,
            x: np.ndarray,
            y: np.ndarray,
            jitter: float = 1e-10,
            fit_options: typing.Optional[dict] = None,
            samples: int = 2000,
            thin: int = 1,
            chains: int = 4,
            adapt_delta: float = 0.8,
            max_treedepth: int = 10,
            seed: typing.Optional[int] = None,
            permute: bool = True,
            init_method: typing.Union[dict, str, int] = 'stable',
    ) -> typing.Union[np.ndarray, dict]:
        """ Sample model hyperparameters using Hamiltonian Monte-Carlo

        Parameters
        ----------
        x : ndarray
            (n,d) array with each row representing a sample point in
            d-dimensional space.
        y : ndarray
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
            x,
            y,
            jitter,
            fit_options,
            samples,
            thin,
            chains,
            adapt_delta,
            max_treedepth,
            seed,
            permute,
            init_method,
        )
        return parameters, info

    def map(
            self,
            x: np.ndarray,
            y: np.ndarray,
            jitter: float = 1e-10,
            fit_options:  typing.Optional[dict] = None,
            init_method: typing.Union[dict, str, int] = 'stable',
            algorithm: str = 'Newton',
            restarts: int = 10,
            max_iter: int = 250,
            seed: typing.Optional[int] = None,
    ) -> typing.Union[np.ndarray, dict]:
        """ Identify model parameters using log-likelihood optimisation

        Parameters
        ----------
        x : ndarray
            (n,d) array with each row representing a sample point in
            d-dimensional space.
        y : ndarray
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
            x,
            y,
            jitter,
            fit_options,
            init_method,
            algorithm,
            restarts,
            max_iter,
            seed
        )
        return parameters, info
