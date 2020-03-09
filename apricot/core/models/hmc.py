import typing
import numpy as np

from apricot.core import utils

def run_hmc(
        interface : 'apricot.core.models.interface.Interface',
        x : np.ndarray,
        y : np.ndarray,
        jitter : float = 1e-10,
        fit_options : typing.Optional[dict] = None,
        samples : int = 2000,
        thin : int = 1,
        chains : int = 4,
        adapt_delta : float = 0.8,
        max_treedepth : int = 10,
        seed : typing.Optional[int] = None,
        permute : bool = True,
        init_method : str = 'stable',
):
    """Run Stan's HMC algorithm for the provided model.

    This is the model interface to Stan's implementation of the No-U-Turn
    Sampler (NUTS) via pyStan.

    Parameters
    ----------
    interface : instance of models.interface.Interface
        Interface to the desired Stan model
    x : ndarray
        (n,d) array with each row representing a sample point in d-dimensional
        space
    y : ndarray
        (n,) array of responses corresponding to each row of x
    jitter : float, optional
        Stability jitter. Default = 1e-10.
    fit_options : dict, optional
    samples : int, optional
        Number of samples to draw from the posterior (accounting for the number
        of chains, warmup and thinning). Default = 2000
    thin : int
        If > 1, keep only every thin samples. Default = 1
    chains : int
        The number of independent chains to draw samples from. Default = 4
    adapt_delta : float < 1
        Adapt_delta control parameter to the sampler. Default = 0.8
    max_treedepth : int
        Maximum sample tree depth control parameter. Default = 10
    seed : {int32, None}
        Seed for numpy's random state. Also used to initialise pyStan.
        Default = None
    permute : bool, optional
        If True, permute the samples
    init_method : {'stable', 0, 'random'}
        String determining the initialisation method for each chain. Default =
        'stable'

    Returns
    -------
    samples : ndarray
        Array of sampled hyperparameters
    info : dict
        Sampling information
    """

    if seed is None:
        seed = utils.random_seed()

    # make the data dictionary
    data = interface.make_pystan_dict(x, y, jitter, fit_options, seed=seed)

    # get the appropriate inits
    init = interface.get_init(init_method, data)

    # clone the init for each chain
    inits = [init] * chains

    control = {
        'adapt_delta' : adapt_delta,
        'max_treedepth' : max_treedepth,
    }

    opts = {
        'data':data,
        'init':inits,
        'control':control,
        'pars':interface._pars_to_sample,
        'chains':chains,
        'iter':int(samples * thin / chains * 2.0),
        'thin':thin,
        'seed':seed,
    }
    result = interface.pystan_model.sampling(**opts)

    samples, info = _hmc_post_internal(result, permute=permute, seed=seed)

    # store the seed and inits for debug purposes
    info['seed'] = seed
    info['inits'] = inits

    # run the checks
    info['passed_rhat'] = _check_rhat(info['rhat'])
    info['passed_divergences'] = _check_divergent(info['divergent'])
    info['passed_saturation'] = _check_tree_saturation(
        info['excess_treedepth'],
        info['max_treedepth']
    )
    info['passed_ebfmi'] = _check_ebfmi(info['e_bfmi'])

    return samples, info

def _hmc_post_internal(
        result : dict,
        permute : bool = True,
        seed : typing.Optional[int] = None,
):

    # retrieve the Rhat positions and number of output variables
    rhat_pos = result.summary()['summary_colnames'].index('Rhat')
    neff_pos = result.summary()['summary_colnames'].index('n_eff')
    rows = result.summary()['summary_rownames'].tolist()
    nrows = len(rows)

    # retrieve the sampler parameters and the number of chains
    sampler_params = result.get_sampler_params(inc_warmup=False)
    nchains = len(sampler_params)

    # extract the raw samples for the requested model parameters
    raw_samples = result.extract(permuted=False)
    iterations = raw_samples.shape[0]

    # allocate containers for results
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
        rhat[param] = result.summary()['summary'][:,rhat_pos][idx]
        n_eff[param] = result.summary()['summary'][:,neff_pos][idx]
 
        # slice out the samples
        for chain in range(nchains):
            a = chain*iterations
            b = (chain+1) * iterations
            samples[a : b, idx] = raw_samples[:, chain, idx]

    # get the max treedepth
    max_td = result.stan_args[0]['control']['max_treedepth']

    # extract the sampler parameters
    for chain in range(nchains):
        a = chain*iterations
        b = (chain+1) * iterations
        chain_id[a: b] = chain + 1
        excess_treedepth[a: b] = max_td - sampler_params[chain]['treedepth__']
        divergent[a: b] = sampler_params[chain]['divergent__']
        e_bfmi[chain] = _calc_ebfmi(sampler_params[chain]['energy__'])

    # permute on request, using random seed if provided
    if permute:
        tup = _permute_aligned((samples, chain_id, excess_treedepth, divergent),seed)
        samples, chain_id, treedepth, divergent = tup

    # make info dictionary
    info = {
        'method':'hmc',
        'colnames':rows,
        'sample_chain_id':chain_id,
        'max_treedepth':max_td,
        'excess_treedepth':excess_treedepth,
        'divergent':divergent,
        'rhat':rhat,
        'n_eff':n_eff,
        'e_bfmi':e_bfmi,
    }
    return samples, info

def _same_lengths(arrays : typing.List[np.ndarray]):
    """Check if all supplied arrays are the same shape in dimension 0"""
    n = None
    for a in arrays:
        if n is None:
            n = a.shape[0]
        if a.shape[0] != n:
            return False
    return True

def _permute_aligned(
        arrays : typing.List[np.ndarray],
        seed : typing.Optional[int] =  None,
):
    """ Permute arrays, retaining row alignment.

    Permute each array in arrays along axis 0 using the same
    permuted indices. This ensures the permuted arrays are still aligned.

    Parameters
    ----------
    arrays : iterable container of ndarray

    Returns
    -------
    permuted arrays : tuple of ndarray
    """
    if not _same_lengths(arrays):
        raise ValueError('Not all provided arrays have identical shapes in axis 0.')
    n = arrays[0].shape[0]
    if seed:
        np.random.seed(seed)
    index = np.random.permutation(n)
    return (array[index] for array in arrays)

def _calc_ebfmi(energy : np.ndarray):
    """ Expected Bayesian Fraction of Missing Information. """
    tmp = np.sum((energy[1:] - energy[:-1])**2) / energy.shape[0]
    return tmp / np.var(energy)

def _check_tree_saturation(excess_treedepth : np.ndarray, max_treedepth : int):
    """ Check for incidences of tree depth saturation. """
    passed = True
    saturations = excess_treedepth[excess_treedepth == 0].shape[0]
    if saturations > 0:
        passed = False
    return passed

def _check_ebfmi(ebfmi):
    """ Check that ebfmi > 0.2. """
    passed = True
    for i,tmp in enumerate(ebfmi < 0.2):
        if tmp:
            passed = False
    return passed

def _check_divergent(divergent : np.ndarray):
    """ Check for divergences. """
    passed = True
    ndivergent = np.sum(divergent)
    if ndivergent > 0:
        passed = False
    return passed

def _check_rhat(rhat):
    """ Check rhat < 1.1. """
    passed = True
    for r in rhat:
        if rhat[r] > 1.1:
            passed = False
    return passed
