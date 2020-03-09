import typing
import functools
import re
import numpy as np

from apricot.core import utils

def slice_in_order(
        array : np.ndarray,
        target : str,
        colnames : typing.List[str],
):
    """ Slice target columns of array corresponding to colnames.

    Automatically strips brackets from colnames such that (for example) 'ls[0],
    ls[1], ls[2]' will match with 'ls', and the columns will be extracted
    in order.
    """
    if any((match(col, target) for col in colnames)):
        return array[:, np.array([match(col, target) for col in colnames])]
    else:
        return None

def match(s : str, target : str):
    """ Strips off any brackets from string x and checks if it is equivalent
    to string target. """
    s_stripped = re.sub('\[.*\]', '', s)
    return s_stripped == target

def param_to_2dfarray(arr : np.ndarray):
    """ Convert not necessarily 1D array to 2D F-ordered array. """
    return np.atleast_1d(arr).reshape(1, -1, order='F')

@utils.maybe
def hmc_glue(
        interface : 'apricot.core.models.interface.Interface',
        samples : np.ndarray,
        info : dict,
):
    """ Formatting for hyperparameters obtained via Stan using HMC.

    Slices out the columns of samples required by the model in order to create
    a dictionary of named hyperparameters.

    Parameters
    ----------
    interface : instance of apricot.core.models.Interface
        Interface to the pystan model used to run the sampler.
    samples : ndarray
        Array of samples obtained via HMC.
    info : dict
        Dictionary of diagnostic information supplied by interface.hmc

    Returns
    -------
    hyperparameters : dict
        dictionary of named hyperparameter samples.
    """

    required_parameters = interface.theta + interface.beta + interface.xi + ['lp__']
    hyperparameters = {}
    
    # iterate over the parameters required by the model
    for parameter in required_parameters:
        hyperparameters[parameter] = slice_in_order(samples, parameter, info['colnames'])
    
    # if the model is deterministic, manually add xi (which is zero)
    if (interface.noise_type[0] == 'deterministic') & ('xi' in required_parameters):
        hyperparameters['xi'] = np.full(
            (samples.shape[0], 1),
            interface.noise_type[1],
            order='F',
            dtype=np.float64
        )
    return hyperparameters

@utils.maybe
def mle_glue(
        interface : 'apricot.core.models.interface.Interface',
        opt_result : typing.Dict[str, typing.Any],
        info : dict,
):
    """ Formatting for hyperparameters obtained via Stan using MLE.

    Parameters
    ----------
    interface : instance of apricot.core.models.Interface
        Interface to the pystan model used to run the sampler.
    opt_result : dict
        Dictionary of optimised hyperparameters.
    info : dict
        Dictionary of diagnostic information supplied by interface.mle

    Returns
    -------
    hyperparameters : dict
        Dictionary of (appropriately formatted) named hyperparameters.
    """

    required_parameters = interface.theta + interface.beta + interface.xi
    hyperparameters = {}
    
    for parameter in required_parameters:
        if parameter in opt_result:
            hyperparameters[parameter] = param_to_2dfarray(opt_result[parameter])
            
    # if the model is deterministic, we need to manually add xi
    if (interface.noise_type[0] == 'deterministic') & ('xi' in required_parameters):
        hyperparameters['xi'] = np.full(
            (1, 1),
            interface.noise_type[1],
            order='F',
            dtype=np.float64
        )    
    return hyperparameters
