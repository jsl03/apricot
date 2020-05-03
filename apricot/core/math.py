"""

DOCSTRING

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import Tuple
import numpy as np  # type: ignore


def integrate_mixture(
        means: np.ndarray,
        variances: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """ Calculate the first two moments of a mixture of Gaussians with uniform
    weights.

    Given n means and variances, compute E[y] and E[(y - E[y])^2]:
    E[y] = mu = mean(means)
    E[(y - E[y])^2] = sigma_sq = mean(means^2 + variances) - mu^2

    Currently faster than using an accumulator in a loop due to numpy.mean
    being speedy when operating on arrays.
    """
    mu = np.mean(means, axis=1)
    variance = np.mean((means**2 + variances), axis=1) - mu**2
    return mu, variance
