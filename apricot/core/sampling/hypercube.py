# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import Optional, Mapping, Any, Callable
import numpy as np  # type: ignore
from apricot.core.sampling.sobol import sobol
from apricot.core.sampling.sobol import sobol_scatter
from apricot.core.sampling.lhs import lhs
from apricot.core.sampling.lhs import optimised_lhs
from apricot.core.sampling.lhs import mdurs
from apricot.core.sampling.factorial import factorial
from apricot.core.utils import set_seed


def urandom(sample_size: int, dimensions: int, seed: Optional[int] = None):
    """ Uniform random sample.

    Parameters
    ----------
    n : int
        Number of random vectors to draw.
    d : int
        Dimension of the random vectors.

    Returns
    -------
    urvs : ndarray
        (n, d) array of n random d-dimensional vectors drawn uniformly at
        random.
    """
    # pylint: disable=no-member
    set_seed(seed)
    return np.random.random((sample_size, dimensions))


# TODO strengthen type of methods here; all accept two ints as the first
# positional argument
METHODS: Mapping[str, Callable] = {
    'urandom': urandom,
    'lhs': lhs,
    'olhs': optimised_lhs,
    'mdurs': mdurs,
    'sobol': sobol,
    'randomised_sobol': sobol_scatter,
    'factorial': factorial,
}


def sample_hypercube(
        sample_size: int,
        dimensions: int,
        method: str,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    """ Unified interface to obtaining uniform random variables on [0,1]^d.

    Generate n d-dimensional vectors using the method 'method'.

    Parameters
    ----------
    sample_size : int
        The number of random variables to draw.
    dimensions : int
        The dimension of the random variables to draw.
    method : {'urandom', 'lhs', 'olhs', 'mdurs', 'sobol', 'randomised_sobol'}
        String matching one of the following available methods:
        * 'urandom' : Uniform random. The cheapest and most simple method is
            recommended for very large sample sizes or if the execution time of
            this function is important, for example inside loops.
        * 'lhs' : Latin hypercube sample. Has more consistent space filling
            properties than 'urandom' and is not as expensive as the either
            'olhs', 'mdurs' or 'sobol'.
        * 'olhs' : Optimised Latin hypercube sample. Generates many Latin-
            Hypercube designs and picks the best according to a specified
            criteria. Criteria defaults to maximin criterion of [1]_.
        * 'mdurs' : Multi-dimensionally uniform random sample [2]_. 'mdurs' is
            expensive but is both "random" and possesses good space filling
            properties. Not recommended for n > 50 unless run time is not a
            priority.
        * 'sobol' : Sobol sequence [3]_. Sobol sequences are both reasonably
            cheap to compute and have good uniform space filling properties.
            However, this implementation of the algorithm is deterministic so
            will provide the same set of points if run repeatedly with the same
            arguments.
        * 'randomised_sobol' : Sobol sequence [2]_ with randomization.
            Generates a sobol sequence, applies uniform [-1,1]^d perturbation
            to each of the generated points, and then modulates the grid to be
            on [0,1]^d.
        * 'factorial' : factorial (cartesian product) grid. Currently only
            supports sample sizes of a**d <= n where n is the requested
            number of samples, d is the number of dimensions and a is the
            returned number of samples. Future versions should allow variable
            numbers of points in each input dimension.
    seed : {None, int32}, optional
        Random seed. Ignored if the algorithm is not randomised. If None,
        an arbitrary random seed will be used. Default = None.
    options : {None, dict}, optional
        Dictionary of additional options to supply to the sampling algorithm.
        Default = None.

    Returns
    -------
    urvs : ndarray
        (sample_size, dimensions) array of independent uniform random variables
        scaled between 0 and 1.

    References
    ----------
    [1] Morris, M.D. and Mitchell, T.J., 1995. Exploratory designs for
    computational experiments. Journal of statistical planning and inference,
    43(3), pp.381-402.

    [2] Deutsch, J.L. and Deutsch, C.V., 2012. Latin hypercube sampling with
    multidimensional uniformity. Journal of Statistical Planning and
    Inference, 142(3), pp.763-772.

    [3] Sobol, I.M., 1976. Uniformly distributed sequences with an additional
    uniform property. USSR Computational Mathematics and Mathematical
    Physics, 16(5), pp.236-242.
    """
    method = method.lower()
    if options is None:
        options = {}
    try:
        return METHODS[method](sample_size, dimensions, seed=seed, **options)
    except KeyError:
        msg = 'method must be one of {0}.'.format(METHODS.keys())
        raise ValueError(msg)
