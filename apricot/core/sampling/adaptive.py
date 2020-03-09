import numpy as np

from apricot.core.sampling.hypercube import sample_hypercube

def max_entropy(
        emulator,
        n : int,
        pool : int = 200,
        method : str = 'lhs',
):
    """ Maximum Entropy LHS sample

    Given an Emulator instance, select from a candidate pool of LHS experimental
    designs that with the highest posterior differential entropy under the model.

    Parameters
    ----------
    Emulator : instance of the Emulator class
        Model to adaptively sample from.
    n : int
        Number of samples in the adaptive design.
    pool : int, optional
        Size of candidate design pool. Default=200.
    method : {'lhs', 'urandom'}
        Method used to generate the candidate grids:
        * lhs - Latin hypercube sample
        * urandom - Uniform random sample.
        Default = 'lhs'.

    Returns
    -------
    grid : ndarray
        (n, d) array describing the adaptive experimental design from the pool
        featuring highest posterior entropy.
    h0 : float
        Differential entropy of grid.
    """
    d = emulator.d
    h0 = -np.inf
    grid = None
    i = 0
    while i < pool:
        cand = sample_hypercube(n, d, 'lhs')
        h = emulator.entropy(cand)
        if h > h0:
            grid = cand
            h0 = h
        i += 1
    return grid, h0
