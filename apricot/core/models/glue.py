# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import List, Optional, Dict, Any
import re
import numpy as np  # type: ignore
from apricot.core.models import type_aliases as ta


# for satisfying forward type checking
if False:  # pylint: disable=using-constant-test
    import apricot  # pylint: disable=unused-import


def slice_in_order(
        array: np.ndarray,
        target: str,
        colnames: List[str],
) -> Optional[np.ndarray]:
    """ Slice target columns of array corresponding to colnames.

    Automatically strips brackets from colnames such that (for example) 'ls[0],
    ls[1], ls[2]' will match with 'ls', and the columns will be extracted
    in order.
    """
    if any((match(col, target) for col in colnames)):
        return array[:, np.array([match(col, target) for col in colnames])]
    return None


def match(name: str, target: str) -> bool:
    """ Strips off any brackets from string name and checks if it is equivalent
    to string target. """
    s_stripped = re.sub(r'\[.*\]', '', name)
    return s_stripped == target


def param_to_2dfarray(arr: np.ndarray) -> np.ndarray:
    """ Convert not necessarily 1D array to 2D F-ordered array. """
    return np.atleast_1d(arr).reshape(1, -1, order='F')


def hmc_glue(
        interface: 'apricot.core.models.interface.Interface',
        samples: np.ndarray,
        info: Dict[str, Any],
) -> ta.Hyperparameters:
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
    required = (
        interface.theta +
        interface.beta +
        interface.sigma +
        ['lp__']
    )
    hyperparameters = {}
    for parameter in required:
        hyperparameters[parameter] = slice_in_order(
            samples,
            parameter,
            info['colnames']
        )
    # if the model is deterministic, manually add sigma and its value
    if (interface.noise_type[0] == 'deterministic') & ('sigma' in required):
        hyperparameters['sigma'] = np.full(
            (samples.shape[0], 1),
            interface.noise_type[1],
            order='F',
            dtype=np.float64
        )
    return hyperparameters


def map_glue(
        interface: 'apricot.core.models.interface.Interface',
        opt_result: Dict[str, Any]
) -> ta.Hyperparameters:
    """ Formatting for hyperparameters obtained via Stan using map.

    Parameters
    ----------
    interface : instance of apricot.core.models.Interface
        Interface to the pystan model used to run the sampler.
    opt_result : dict
        Dictionary of optimised hyperparameters.
    info : dict
        Dictionary of diagnostic information supplied by interface.map

    Returns
    -------
    hyperparameters : dict
        Dictionary of (appropriately formatted) named hyperparameters.
    """
    required = (
        interface.theta +
        interface.beta +
        interface.sigma
    )
    hyperparameters = {}
    for parameter in required:
        if parameter in opt_result:
            hyperparameters[parameter] = param_to_2dfarray(
                opt_result[parameter]
            )
    # if the model is deterministic, manually add sigma and its value
    if (interface.noise_type[0] == 'deterministic') & ('sigma' in required):
        hyperparameters['sigma'] = np.full(
            (1, 1),
            interface.noise_type[1],
            order='F',
            dtype=np.float64
        )
    return hyperparameters
