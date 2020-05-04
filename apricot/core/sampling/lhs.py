# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import cast, Optional, Union, Mapping, Any
import numpy as np  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from apricot.core.utils import set_seed
from apricot.core.models import type_aliases as ta


def _maximin(pairwise_distances: np.ndarray, p_const: int = 2) -> float:
    """ 'maximin' LHS quality criteria of Morris and Mitchell (1995).

    The maximin LHS optimality criterion of [1]_.

    Parameters
    ----------
    pairwise_distances: ndarray
        vector of pairwise distances for the sample
    p_const: int
        Constant. You should not need to change this; see [1]_.

    Returns
    -------
    phi : float
        maximin criteria evaluated for supplied pairwise distances.

    References
    ----------
    [1] Morris, M.D. and Mitchell, T.J., 1995. Exploratory designs for
    computational experiments. Journal of statistical planning and inference,
    43(3), pp.381-402.
    """
    num_pts = pairwise_distances.shape[0]
    # sort the pairwise distances
    distances_sorted = np.sort(pairwise_distances)
    # initialise the accumulator
    phi = 0
    for i in range(num_pts):
        # count the number of distances smaller than element i
        _d = distances_sorted[i]
        _j = np.array(distances_sorted < distances_sorted[i], dtype=bool).sum()
        # compute the maximin criteria using J and D
        phi += _j*_d**(-p_const)
    return -(phi**(1./p_const))


# dict of criteria usable by optimised_lhs
CRITERIA: Mapping[str, ta.LhsCriteria] = {
    'maximin': _maximin,
}


def lhs(
        sample_size: int,
        dimensions: int,
        seed: Optional[int] = None
) -> np.ndarray:
    """ Latin Hypercube Sample design.

    Generate n stratified samples in d dimensions by drawing samples from a
    latin hypercube.

    'lhs' is faster than both 'mdurs' and 'optimised_lhs' but has less
    consistent uniformity properties, especially in higher numbers of
    dimensions.

    Parameters
    ----------
    sample_size: int
        Number of requested sample points
    dimensions: int
        Number of dimensions to sample in
    seed: {None, int}, optional
        Seed for numpy's random state. If None, an arbitrary seed is generated.
        Default = None.

    Returns
    -------
    sample: ndarray
        (sample_size, dimensions) array of n sample points in d dimensions.
        Results are scaled on [0,1] by default.

    See Also
    --------
    mdurs
    optimised_lhs
    """
    # pylint: disable=no-member
    set_seed(seed)
    slices = np.linspace(0, 1, sample_size + 1)
    urnd = np.random.random((sample_size, dimensions))
    lower = slices[:sample_size]
    upper = slices[1:]
    points = np.empty((sample_size, dimensions), order='C', dtype=np.float64)
    sample = np.empty((sample_size, dimensions), order='C', dtype=np.float64)
    for j in range(dimensions):
        points[:, j] = urnd[:, j] * (upper - lower) + lower
        index = np.random.permutation(range(sample_size))
        sample[:, j] = points[index, j]
    return sample


def mdurs(
        sample_size: int,
        dimensions: int,
        scale_factor: int = 10,
        nearest_k: int = 2,
        measure: str = 'cityblock',
        seed: Optional[int] = None,
) -> np.ndarray:
    """ Multi-Dimensionally Uniform Random Sample.

    Implements the "LHSMDU" algorithm of Deutsch and Deutsch [1]_.

    mdurs is suited to randomised designs of low (n < 50) numbers of samples.
    Though it can be used for larger n, runtime may become an issue as the
    algorithm iterates over individual sample points in a canidate pool rather
    than the sample designs themselves.

    mdurs uses one of scipy's distance measures to maximise dispersion between
    points. Valid distance measures are any supported by
    scipy.spatial.distance.cdist, of which typical choices are:
    * 'cityblock' : L1 distance
    * 'eculidean' : L2 distance
    * 'sqeuclidean' : squared L2 distance

    Parameters
    ----------
    sample_size: int
        Number of requested sample points
    dimensions: int
        Number of dimensions
    scale_factor : int, optional
        Scale factor (default = 10). You should not need to change this; see
        [1]_.
    nearest_k: int, optional
        Number of neighbours used to compute moving average (default = 2).
        You should not need to change this; see [1]_.
    measure: string, optional
        Distance measure to be used. Passed as a method argument to scipy's
        spatial.distance.cdist function. Default = 'cityblock'.
    seed: {None, int}, optional
        Seed for numpy's random state. If None, an arbitrary seed is generated.
        Default = None.

    Returns
    -------
    random_sample: ndarray
        (sample_size, dimensions) array of n sample points in d dimensions.
        Results are scaled on [0,1].

    Notes
    -----
    This algorithm is unusably slow for large n. For n > 50 it is recommended
    to use one of the other sampling algorithms unless the time required to
    generate the sample points is less important than a highly uniform random
    sample.

    References
    ----------
    [1] Deutsch, J.L. and Deutsch, C.V., 2012. Latin hypercube sampling with
    multidimensional uniformity. Journal of Statistical Planning and
    Inference, 142(3), pp.763-772.

    See Also
    --------
    lhs
    optimised_lhs
    scipy.spatial.distance.cdist
    """
    # pylint: disable=no-member
    set_seed(seed)
    n_pool = scale_factor * sample_size
    random_sample = np.random.random((n_pool, dimensions))
    while random_sample.shape[0] > sample_size:
        len_s = random_sample.shape[0]
        distance_matrix = cdist(random_sample, random_sample, metric=measure)
        ret = np.empty(len_s, dtype=np.float64, order='C')
        for i in range(len_s):
            ret[i] = np.mean(np.sort(distance_matrix[i, :])[1:1 + nearest_k])
        random_sample = np.delete(random_sample, np.argmin(ret), axis=0)
    return random_sample


# TODO: fix random seed behaviour + possible refactor
def optimised_lhs(
        sample_size: int,
        dimensions: int,
        iterations: int = 100,
        measure: str = 'euclidean',
        criteria: Union[str, ta.LhsCriteria] = 'maximin',
        options: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
) -> np.ndarray:
    """Optimised Latin Hypercube Sample design.

    Pick a sample from a collection of latin hypercube designs maximising a
    specified criteria, nominally the 'maximin' criteria of Morris and
    Mitchell [1]_.

    optimised_lhs generates a large number of lhs designs, then selects from
    this pool the design best satisfying a specified criteria, which is a
    function of a specified distance measure (or 'metric' - though 'metric'
    is not necessarily a metric in the mathematical sense).

    A valid distance measure is any supported by scipy's cdist, of which
    typical choices are:
        * 'cityblock' : L1 distance
        * 'eculidean' : L2 distance
        * 'sqeuclidean' : squared L2 distance

    Currently supported comparison criteria are:
        * 'maximin' [1]_.

    Parameters
    ----------
    sample_size: int
        Number of requested sample points
    dimensions: int
        Number of dimensions
    iterations: int, optional
        The number of individual designs to compare. The design maximising
        'criteria' after the requested number of iterations will be returned.
        Default = 100.
    measure: str, optional
        Distance measure to be used for comparing designs. References one of
        the measures compatible with scipy's spatial.distance.cdist function.
        Default = 'euclidean'.
    criteria: {str, callable}, optional
        Comparison criteria:
        * 'maximin' - maximin criteria.
        * callable - user supplied function; see below.

    Returns
    -------
    sample: ndarray
        (sample_size, dimensions) array of n sample points in d dimensions.
        Results are scaled on [0,1].

    Notes
    -----
    A user supplied function can be used as a comparison criteria. The supplied
    function accepts a vector of pairwise distances calculated using 'measure'.
    The function should return a quantity intended to be maximised.

    References
    ----------
    [1] Morris, M.D. and Mitchell, T.J., 1995. Exploratory designs for
    computational experiments. Journal of statistical planning and inference,
    43(3), pp.381-402.

    See Also
    --------
    scipy.spatial.distance.cdist
    """
    # pylint: disable=too-many-arguments, too-many-locals, no-member
    set_seed(seed)
    if options is None:
        options = {}
    slices = np.linspace(0, 1, sample_size+1)
    lower = slices[:sample_size]
    upper = slices[1:]
    indices_list = np.arange(sample_size)
    points = np.empty((sample_size, dimensions), order='C', dtype=np.float64)
    sample = np.empty((sample_size, dimensions), order='C', dtype=np.float64)
    tmp = -np.inf
    for _ in range(iterations):
        urnd = np.random.random((sample_size, dimensions))
        for j in range(dimensions):
            points[:, j] = urnd[:, j] * (upper - lower) + lower
            index = np.random.permutation(indices_list)
            sample[:, j] = points[index, j]
        delta = eval_criteria(sample, measure, criteria, options)
        if delta > tmp:
            tmp = delta
            ret = sample.copy()  # is this copy necessary?
    return ret


def eval_criteria(
        arr: np.ndarray,
        measure: str,
        criteria: Union[str, ta.LhsCriteria],
        options: Mapping[str, Any]
) -> float:
    """ Evaluate LHS optimality criteria.

    Parameters
    ----------
    arr : ndarray
        The array for which to evaluate the criteria.
    measure : str
        String describing a scipy.spatial.distance.cdist compatible metric.
    criteria : {str, callable}
        Name of the criteria to use or user supplied function.
    options : dict
        Additional keyword arguments to pass to criteria.

    Returns
    -------
    criterion : float
        Criteria, as evaluated for arr using measure.
    """
    n_pts = arr.shape[0]
    i_x, j_y = np.tril_indices(n_pts, -1)
    dist = cdist(arr, arr, metric=measure)
    dist_vector = dist[i_x, j_y]

    if isinstance(criteria, str):
        if criteria in CRITERIA:
            func = CRITERIA[criteria]
            return func(dist_vector, **options)

    if hasattr(criteria, '__call__'):
        # explicitly cast criteria as a function to satisfy type checker
        func = cast(ta.LhsCriteria, criteria)
        return func(dist_vector, **options)

    msg = (
        'critiera must be either a function(ndarray) -> float'
        ' or one of: {0}'.format(CRITERIA)
    )
    raise ValueError(msg)
