"""Miscellaneous mathematical operations should go here."""

import numpy as np

def integrate_mixture(means, variances):
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
