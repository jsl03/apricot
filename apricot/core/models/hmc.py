"""
Code for fitting a model using Hamiltonian Monte-Carlo.

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import cast, Optional, Union, Dict, Any, List, Sequence, Tuple
import numpy as np  # type: ignore
from apricot.core import utils
from apricot.core.logger import get_logger
from apricot.core.models import type_aliases as ta


# for satisfying forward type checking
if False:  # pylint: disable=using-constant-test
    import apricot  # pylint: disable=unused-import


LOGGER = get_logger()


def run_hmc(  # pylint: disable=too-many-arguments, too-many-locals
        interface: 'apricot.core.models.interface.Interface',
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
    """Run Stan's HMC algorithm for the provided model.

    This is the model interface to Stan's implementation of the No-U-Turn
    Sampler (NUTS) via pyStan.

    Parameters
    ----------
    interface: instance of models.interface.Interface
        Interface to the desired Stan model
    x_data: ndarray
        (n,d) array with each row representing a sample point in d-dimensional
        space
    y_data: ndarray
        (n,) array of responses corresponding to each row of x
    jitter: float, optional
        Stability jitter. Default = 1e-10.
    ls_options: dict, optional
        Options to fine tune the behaviour of the lengthscale prior
        distribution.
    samples: int, optional
        Number of samples to draw from the posterior (accounting for the number
        of chains, warmup and thinning). Default = 2000
    thin: int
        If > 1, keep only every thin samples. Default = 1
    chains: int
        The number of independent chains to draw samples from. Default = 4
    adapt_delta: float < 1
        Adapt_delta control parameter to the sampler. Default = 0.8
    max_treedepth: int
        Maximum sample tree depth control parameter. Default = 10
    seed: {int32, None}
        Seed for numpy's random state. Also used to initialise pyStan.
        Default = None
    permute: bool, optional
        If True, permute the samples
    init_method: {'stable', 0, 'random'}
        String determining the initialisation method for each chain. Default =
        'stable'

    Returns
    -------
    samples: ndarray
        Array of sampled hyperparameters
    info: dict
        Sampling information
    """
    if seed is None:
        seed = utils.random_seed()
    data = interface.make_pystan_dict(
        x_data,
        y_data,
        jitter,
        ls_options=ls_options,
        seed=seed
    )
    init = interface.get_init(init_method, data)
    init_sampler = assign_init(init, chains)
    control = {
        'adapt_delta': adapt_delta,
        'max_treedepth': max_treedepth,
    }
    opts = {
        'data': data,
        'init': init_sampler,
        'control': control,
        'pars': interface.pars_to_sample,
        'chains': chains,
        'iter': int(samples * thin / chains * 2.0),
        'thin': thin,
        'seed': seed,
    }
    result = interface.pystan_model.sampling(**opts)
    samples, info = _hmc_post_internal(result, permute=permute, seed=seed)
    info['seed'] = seed
    info['init'] = init_sampler
    info['passed_rhat'] = check_rhat(info['rhat'])
    info['passed_divergences'] = check_divergent(info['divergent'])
    info['passed_saturation'] = check_tree_saturation(info['excess_treedepth'])
    info['passed_ebfmi'] = check_ebfmi(info['e_bfmi'])
    return samples, info


def _hmc_post_internal(  # pylint: disable=too-many-locals
        result,
        permute: bool = True,
        seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """

    Parameters
    ----------
    result:
        Raw pyStan.sampling output.
    permute: bool
        Bool.
    seed : {None, int}, optional
        Random seed.

    Returns
    -------
    samples: ndarray
        Array of sampled hyperparameters.
    info: dict
        Sampling information.
    """
    # retrieving the Rhat positions and number of output variables
    rhat_pos = result.summary()['summary_colnames'].index('Rhat')
    neff_pos = result.summary()['summary_colnames'].index('n_eff')
    rows = result.summary()['summary_rownames'].tolist()
    nrows = len(rows)
    sampler_params = result.get_sampler_params(inc_warmup=False)
    nchains = len(sampler_params)
    raw_samples = result.extract(permuted=False)
    iterations = raw_samples.shape[0]
    rhat = {}
    n_eff = {}
    e_bfmi = np.empty(nchains)
    samples = np.empty((nchains * iterations, nrows))
    chain_id = np.empty(nchains * iterations)
    divergent = np.empty(nchains * iterations)
    excess_treedepth = np.empty(nchains * iterations)
    # get the rhat values and number of effective samples
    for param in rows:
        idx = rows.index(param)
        rhat[param] = result.summary()['summary'][:, rhat_pos][idx]
        n_eff[param] = result.summary()['summary'][:, neff_pos][idx]
        # slice out the samples
        for chain in range(nchains):
            idx_min = chain*iterations
            idx_max = (chain+1) * iterations
            samples[idx_min: idx_max, idx] = raw_samples[:, chain, idx]
    max_td = result.stan_args[0]['control']['max_treedepth']
    # extract the sampler parameters
    for chain in range(nchains):
        _params = sampler_params[chain]
        idx_min = chain*iterations
        idx_max = (chain+1) * iterations
        chain_id[idx_min: idx_max] = chain + 1
        excess_treedepth[idx_min: idx_max] = max_td - _params['treedepth__']
        divergent[idx_min: idx_max] = _params['divergent__']
        e_bfmi[chain] = calc_ebfmi(_params['energy__'])
    if permute:
        samples, chain_id, excess_treedepth, divergent = permute_aligned(
            (samples, chain_id, excess_treedepth, divergent), seed
        )
    info = {
        'method': 'hmc',
        'colnames': rows,
        'sample_chain_id': chain_id,
        'max_treedepth': max_td,
        'excess_treedepth': excess_treedepth,
        'divergent': divergent,
        'rhat': rhat,
        'n_eff': n_eff,
        'e_bfmi': e_bfmi,
    }
    return samples, info


def assign_init(init: ta.InitTypes, chains: int) -> ta.PyStanInitTypes:
    """ If init is single a dictionary, it must be copied for each chain."""
    if isinstance(init, dict):
        return [cast(ta.InitData, init)] * chains
    # if init is not a dict, it is either a string, literal zero, a list of
    # either of the preceeding, or a list of dictionaries. Necessary to make
    # mypy aware of this explicitly
    return cast(
        Union[List[ta.InitData], List[str], List[ta.Zero], str, ta.Zero],
        init
    )


def same_lengths(arrays: Sequence[np.ndarray]) -> bool:
    """Check if all supplied arrays are the same shape in dimension 0"""
    length = None
    for arr in arrays:
        if length is None:
            length = arr.shape[0]
        if arr.shape[0] != length:
            return False
    return True


def permute_aligned(
        arrays: Sequence[np.ndarray],
        seed: Optional[int] = None,
) -> List[np.ndarray]:
    """ Permute arrays, retaining row alignment.

    Permute each array in arrays along axis 0 using the same
    permuted indices. This ensures the permuted arrays are still aligned.

    Parameters
    ----------
    arrays: iterable container of ndarray
        The arrays to permute.

    Returns
    -------
    permuted_arrays: tuple of ndarray
        The permuted arrays.
    """
    if not same_lengths(arrays):
        msg = 'Not all of the provided arrays are the same length in axis 0.'
        raise ValueError(msg)
    length = arrays[0].shape[0]
    if seed:
        np.random.seed(seed)
    index = np.random.permutation(length)
    return [array[index] for array in arrays]


def calc_ebfmi(energy: np.ndarray) -> np.ndarray:
    """ Expected Bayesian Fraction of Missing Information. """
    tmp = np.sum((energy[1:] - energy[:-1])**2) / energy.shape[0]
    return tmp / np.var(energy)


def check_tree_saturation(excess_treedepth: np.ndarray) -> bool:
    """ Check for incidences of tree depth saturation. """
    passed = True
    saturations = excess_treedepth[excess_treedepth == 0].shape[0]
    if saturations > 0:
        passed = False
    return passed


def check_ebfmi(ebfmi) -> bool:
    """ Check that ebfmi > 0.2. """
    passed = True
    if any(ebfmi < 0.2):
        passed = False
    return passed


def check_divergent(divergent: np.ndarray) -> bool:
    """ Check for divergences. """
    passed = True
    ndivergent = np.sum(divergent)
    if ndivergent > 0:
        passed = False
    return passed


def check_rhat(rhat: Dict[str, Any]) -> bool:
    """ Check rhat < 1.1 for all parameters """
    passed = True
    for param in rhat:
        if rhat[param] > 1.1:
            passed = False
    return passed
