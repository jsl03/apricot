# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import Union
import numpy as np  # type: ignore
from apricot.core.sampling.hypercube import sample_hypercube


def max_entropy(
        emulator,
        sample_size: int,
        pool: int = 200,
        method: str = 'lhs',
) -> Union[np.ndarray, float]:
    """ Maximum Entropy LHS sample

    Given an Emulator instance, select from a candidate pool of LHS
    experimental designs that with the highest posterior differential entropy
    under the model.

    Parameters
    ----------
    emulator: instance of the Emulator class
        Model to adaptively sample from.
    sample_size: int
        Number of samples in the adaptive design.
    pool: int, optional
        Size of candidate design pool. Default=200.
    method: {'lhs', 'urandom'}
        Method used to generate the candidate grids:
        * lhs - Latin hypercube sample
        * urandom - Uniform random sample.
        Default = 'lhs'.

    Returns
    -------
    sample: ndarray
        (n, d) array describing the adaptive experimental design from the pool
        featuring highest posterior entropy.
    entropy_max: float
        Differential entropy of returned sample.
    """
    grid_dimension = emulator.index_dimension
    max_observed_entropy = -np.inf
    sample = None
    i = 0
    while i < pool:
        candidate_grid = sample_hypercube(sample_size, grid_dimension, method)
        entropy = emulator.entropy(candidate_grid)
        if entropy > max_observed_entropy:
            sample = candidate_grid
            max_observed_entropy = entropy
        i += 1
    return sample, max_observed_entropy
