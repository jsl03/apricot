import numpy as np
from scipy.spatial.distance import cdist
from apricot.core.utils import set_seed

def _maximin(v, p=2):
    """'maximin' LHS quality criteria of Morris and Mitchell (1995).

    The maximin LHS optimality criterion of [1]_.

    Parameters
    ----------
    v : ndarray
        vector of pairwise distances for the sample

    Returns
    -------
    phi : float
        maximin criteria evaluated for v

    References
    ----------
    [1] Morris, M.D. and Mitchell, T.J., 1995. Exploratory designs for
    computational experiments. Journal of statistical planning and inference,
    43(3), pp.381-402.
    """
    m = v.shape[0]
    # sort the pairwise distances
    v_sorted = np.sort(v)
    # initialise the accumulator
    phi = 0
    for i in range(m):
        # count the number of elements of v smaller than the element i
        D = v_sorted[i]
        J = np.array(v_sorted < v_sorted[i], dtype=bool).sum()
        # compute the maximin criteria using J and D
        phi += J*D**(-p)
    return -(phi**(1./p))

# dict of criteria usable by optimised_lhs
_CRITERIA = {
    'maximin': _maximin,
}

def lhs(n, d, seed=None):
    """Latin Hypercube Sample design.

    Generate n stratified samples in d dimensions by drawing samples from a
    latin hypercube.

    'lhs' is faster than both 'mdurs' and 'optimised_lhs' but has less consistent
    uniformity properties, especially in higher numbers of dimensions.

    Parameters
    ----------
    n : int
        Number of requested sample points
    d : int
        Number of dimensions to sample in

    Returns
    -------
    s : ndarray
        (n,d) array of n sample points in d dimensions. Results are scaled on
        [0,1] by default.

    See Also
    --------
    mdurs
    optimised_lhs
    """
    set_seed(seed)
    slices = np.linspace(0,1,n+1)
    urnd = np.random.random((n,d))
    l = slices[:n]
    u = slices[1:]
    points = np.empty((n,d), order='C', dtype=np.float64)
    s = np.empty((n,d), order='C', dtype=np.float64)
    for j in range(d):
        points[:,j] = urnd[:,j] * (u-l) + l
        index = np.random.permutation(range(n))
        s[:,j] = points[index, j]
    return s

def mdurs(n, d, scale_factor=10, k=2, measure='cityblock', seed=None):
    """Multi-dimensionally uniform random sample.

    Implements the "LHSMDU" algorithm of Deutsch and Deutsch [1]_.

    mdurs is suited to randomised designs of low (n < 50) numbers of samples.
    Though it can be used for larger n, runtime may become an issue as the
    algorithm iterates over individual sample points in a canidate pool rather
    than the sample designs themselves.

    Parameters
    ----------
    n : int
        Number of requested sample points
    d : int
        Number of dimensions
    scale_factor : int, optional
        Scale factor (default = 10). See citation: you should not need to
        change this.
    k : int, optional
        Number of neighbours used to compute moving average (default = 2).
        See citation: you should not need to change this.
    measure : string, optional
        Distance measure to be used (default = 'cityblock'). References one of
        the metrics compatible with scipy's cdist function.

    Returns
    -------
    S : ndarray
        (n,d) array of n sample points in d dimensions.
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
    set_seed(seed)
    nr = scale_factor * n
    S = np.random.random((nr, d))
    while S.shape[0] > n:
        l = S.shape[0]

        # old versions of scipy don't support the out argument in cdist!
        # D = np.empty((l, l), np.float64, order='C')
        D = cdist(S, S, metric=measure)
        ret = np.empty(l, dtype=np.float64, order='C')
        for i in range(l):
            ret[i] = np.mean(np.sort(D[i,:])[1:1 + k])
        S = np.delete(S, np.argmin(ret), axis=0)
    return S

def optimised_lhs(n, d, iterations=100, measure='euclidean', criteria='maximin',
                  options=None, seed=None):
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
    n : int
        Number of requested sample points
    d : int
        Number of dimensions
    iterations : int, optional
        The number of individual designs to compare. The design maximising
        'criteria' after the requested number of iterations will be returned.
    measure : str, optional
        Distance measure to be used for comparing designs (default =
        'euclidean'). References one of the measures compatible with scipy's
        spatial.distance.cdist function (called a 'metric' in scipy).
    criteria : str, optional
        Comparison criteria:
        * 'maximin' - maximin criteria.
        * callable - user supplied function; see below.

    Notes
    -----
    A user supplied function can be used as a comparison criteria. The supplied
    function accepts a vector of pairwise distances calculated using 'measure'.
    The function should return a quantity intended to be maximised.

    Returns 
    -------
    s : ndarray
        (n,d) array of n sample points in d dimensions. Results are scaled on
        [0,1] by default.

    References
    ----------
    [1] Morris, M.D. and Mitchell, T.J., 1995. Exploratory designs for
    computational experiments. Journal of statistical planning and inference,
    43(3), pp.381-402.

    See Also
    --------
    scipy.spatial.distance.cdist : documentation for different measures.
    evalCriteria : evaluates 'criteria' for 'measure'.
    """

    set_seed(seed)

    if options is None:
        options = {}

    # precompute slices and slice indices
    slices = np.linspace(0,1,n+1)
    l = slices[:n]
    u = slices[1:]
    indices_list = np.arange(n)

    points = np.empty((n,d), order='C', dtype=np.float64)
    s = np.empty((n,d), order='C', dtype=np.float64)
    tmp = -np.inf

    for i in range(iterations):

        # draw the uniform random numbers and set strata
        urnd = np.random.random((n,d))
        for j in range(d):
            points[:,j] = urnd[:,j] * (u-l) + l

            # permute the points
            index = np.random.permutation(indices_list)
            s[:,j] = points[index, j] 

        # evaluate the criteria
        delta = eval_criteria(s, measure, criteria, options)

        # if the design is an improvement over the current best, keep it
        if delta > tmp:
            tmp = delta
            ret = s.copy()

    return ret

def eval_criteria(arr, measure, criteria, options):
    """Evaluate LHS optimality criteria."""
    n = arr.shape[0]
    ix, jy = np.tril_indices(n, -1)
    dist = cdist(arr, arr, metric=measure)
    dist_vector = dist[ix,jy]

    if criteria in _CRITERIA:
        return _CRITERIA[criteria](dist_vector, **options)

    elif hasattr(criteria, '__call__'):
        return criteria(dist_vector, **options)

    else:
        raise ValueError('Critiera must be one of {keys} or a callable function'\
                         .format(keys = _CRITERIA.viewkeys()))
