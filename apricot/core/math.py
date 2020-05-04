"""

DOCSTRING

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import Tuple, Optional
import numpy as np  # type: ignore


def mad(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """ Median absolute deviation.

    Parameters
    ----------
    arr : ndarray
        The array on which to calculate the MAD.
    axis : {int, None}, optional
        If provided, calculated the MAD along the specified axis.

    Returns
    -------
    mad : {float, ndarray}
        The MAD, calculated along the specified axis if provided.
    """
    deviation = np.abs(arr - np.mean(arr, axis=axis))
    return np.median(deviation, axis=axis)


def integrate_mixture(
        means: np.ndarray,
        variances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculate the first two moments of a mixture of Gaussians with uniform
    weights.

    Given n means and variances, compute E[y] and E[(y - E[y])^2]:
    E[y] = mu = mean(means)
    E[(y - E[y])^2] = sigma_sq = mean(means^2 + variances) - mu^2

    Currently faster than using an accumulator in a loop due to numpy.mean
    being speedy when operating on arrays.
    """
    expectation = np.mean(means, axis=1)
    variance = np.mean((means**2 + variances), axis=1) - expectation**2
    return expectation, variance
