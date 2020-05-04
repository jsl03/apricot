# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import Optional
import numpy as np  # type: ignore
from apricot.core.logger import get_logger
logger = get_logger()


def factorial(
        sample_size: int,
        dimensions: int,
        seed: Optional[int] = None,
) -> np.ndarray:
    """ Factorial sample design

    Factorial will round down the requested number of samples to the nearest
    factorial number, n_prime, if n is not a factorial number for the specified
    number of dimensions.

    Parameters
    ----------
    n : int
        The requested number of samples
    d : int
        The dimension of the samples.
    seed : {None, int}
        Random seed. This keyword argument is ignored by this sampling
        method as the algorithm is deterministic: it is present for consistency
        with other methods.
    Returns
    -------
    samples : ndarray
        (n_prime, d) array of samples
    """
    n_valid, next_largest = _power_d_less_n(sample_size, dimensions)
    if sample_size != next_largest:
        logger.info(
            'Next largest factorial sample size in %(d)s dimensions is %(n)s.',
            {'d': dimensions, 'n': next_largest}
        )
        logger.warning(
            'Factorial dimension rounded down to %(d)s. Sample size is %(n)s.',
            {'d': n_valid, 'n': n_valid**dimensions}
        )
    else:
        n_valid += 1
    vecs = (np.linspace(0, 1, n_valid) for _ in range(dimensions))
    return cartesian(*vecs)


def cartesian(*args: np.ndarray):
    """Cartesian product of a sequence of arrays"""
    n_args = len(args)
    return np.array(np.meshgrid(*args)).T.reshape(-1, n_args, order='C')


def _power_d_less_n(n: int, d: int):
    """ find a such that d**a < n"""
    # pylint: disable= invalid-name
    a = 1
    while True:
        if a ** d < n:
            a += 1
        else:
            return a-1, a**d
