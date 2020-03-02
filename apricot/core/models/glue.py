import functools
import re
import numpy as np
from collections import namedtuple
from apricot.core.utils import maybe

def slice_in_order(array, target, colnames):
    """ Slice target columns of array corresponding to colnames

    Automatically strips brackets from colnames such that (for example) 'ls[0],
    ls[1], ls[2]' will match with 'ls', and the columns will be extracted
    in order.
    """
    if any((match(col, target) for col in colnames)):
        return array[:, np.array([match(col, target) for col in colnames])]
    else:
        return None

def match(x, target):
    """ Strips off any brackets from string x and checks if it is equivalent
    to string target."""
    x_ = re.sub('\[.*\]', '', x)
    return x_ == target

def param_to_2dfarray(param):
    """ Convert not necessarily 1D array to 2D F-ordered array"""
    return np.atleast_1d(param).reshape(1, -1, order='F')

@maybe
def hmc_glue(interface, samples, info):
    """Formatting for hyperparameters obtained via Stan using HMC"""

    required_parameters = interface.theta + interface.beta + interface.xi + ['lp__']
    hyperparameters = {}
    
    # iterate over the parameters required by the model
    for parameter in required_parameters:
        hyperparameters[parameter] = slice_in_order(samples, parameter, info['colnames'])
    
    # if the model is deterministic, we need to manually add xi
    if (interface.noise_type[0] == 'deterministic') & ('xi' in required_parameters):
        hyperparameters['xi'] = np.full(
            (samples.shape[0], 1),
            interface.noise_type[1],
            order='F',
            dtype=np.float64
        )
    return hyperparameters

@maybe
def mle_glue(interface, opt_result, info):
    """Formatting for hyperparameters obtained via Stan using MLE"""

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
