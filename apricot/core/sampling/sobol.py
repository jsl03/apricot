"""
This module is a stripped down and slightly modified version of an old version
of the "sobol_seq" package. All credits go to the original code, which is
available online at https://github.com/naught101/sobol_seq.

The sobol_seq package is in turn based on original contributions as follows:
    - Original FORTRAN77 version of i4_sobol by Bennett Fox.
    - MATLAB version by John Burkardt.
    - PYTHON version by Corrado Chisari
    - Original Python version of is_prime by Corrado Chisari
    - Original MATLAB versions of other functions by John Burkardt.
    - PYTHON versions by Corrado Chisari
    - Original code available at:
        http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html

This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import Optional
import numpy as np  # type: ignore
import six
from apricot.core.utils import set_seed


def sobol_scatter(
        sample_size: int,
        dimensions: int,
        seed: Optional[int] = None,
        generator_seed: int = 1,
        skip: int = 0,
) -> np.ndarray:
    """ Generate n d-dimensional quasi-random vectors using the Sobol sequence,
    with additive uniform randomisation.

    Implements i4sobol_generate from the sobol_seq package but without the
    ability to change the dimension of the sequence after it has been
    initialised, and with randomization applied to the generated points.

    Parameters
    ----------
    sample_size: int
        The number of random vectors to retrieve.
    dimensions: int
        The dimension of the random vectors
    seed: {None, int32}
        Seed for numpy's random state. If None, an arbitrary seed will be used.
        Default = None.
    generator_seed : int
        Seed for the Sobol sequence generator. Default = 1.
    skip: int
        Skip every this number of generated points. Default = 0.

    Returns
    -------
    samples : ndarray
        (sample_size, dimensions) array consisting of the requested number of
        quasi-random vectors scaled between [0, 1] in the requested number of
        dimensions.

    Notes
    -----
    This function implements i4sobol_generate from the sobol_seq using a
    generator instead of global variable declarations. See the original
    source code at https://github.com/naught101/sobol_seq for more details,
    and the documentation of i4_sobol2 for a full list of references.

    TODO: implement ability to resume sampling from an existing Sobol sequence.

    References
    ----------
    [1] Sobol, I.M., 1976. Uniformly distributed sequences with an additional
    uniform property. USSR Computational Mathematics and Mathematical
    Physics, 16(5), pp.236-242.

    See Also
    --------
    sobol : Sobol sequence without randomisation.
    i4_sobol2 : Sobol sequence generator.

    """
    # pylint: disable=no-member
    set_seed(seed)
    sample = np.empty((sample_size, dimensions), dtype=np.float64)
    seq_generator = i4_sobol2(
        dimensions,
        generator_seed=generator_seed,
        skip=skip
    )
    for j in range(sample_size):
        sample[j, :] = six.next(seq_generator)
    sample += np.random.random(size=sample.shape)
    while (sample < 0).any() | (sample > 1).any():
        sample[sample > 1] -= 1
        sample[sample < 0] += 1
    return sample


def sobol(
        sample_size: int,
        dimensions: int,
        seed: Optional[int] = None,
        generator_seed: int = 1,
        skip: int = 0,
) -> np.ndarray:
    """ Generate n length d quasi-random vectors from the Sobol sequence.

    Generate n length d quasi-random vectors using the Sobol sequence [1]_.

    Implements i4sobol_generate from the sobol_seq package but without the
    ability to change the dimension of the sequence after it has been
    initialised.

    Parameters
    ----------
    sample_size: int
        The number of random vectors to retrieve.
    dimensions: int
        The dimension of the random vectors
    seed: {None, int32}
        Seed for numpy's random state. If None, an arbitrary seed will be used.
        Default = None.
    generator_seed : int
        Seed for the Sobol sequence generator. Default = 1.
    skip: int
        Skip every this number of generated points. Default = 0.

    Returns
    -------
    samples : ndarray
        (sample_size, dimensions) array consisting of the requested number of
        quasi-random vectors scaled between [0, 1] in the requested number of
        dimensions.

    Notes
    -----
    This function implements i4sobol_generate from the sobol_seq using a
    generator instead of global variable declarations. See the original
    source code at https://github.com/naught101/sobol_seq for more details,
    and the documentation of i4_sobol2 for a full list of references.

    TODO: implement ability to resume sampling from an existing Sobol sequence.

    References
    ----------
    [1] Sobol, I.M., 1976. Uniformly distributed sequences with an additional
    uniform property. USSR Computational Mathematics and Mathematical
    Physics, 16(5), pp.236-242.

    See Also
    --------
    sobol_scatter : Sobol sequence with additive randomisation.
    i4_sobol2 : Sobol sequence generator.
    """
    set_seed(seed)
    sample = np.empty((sample_size, dimensions), dtype=np.float64)
    seq_generator = i4_sobol2(
        dimensions,
        generator_seed=generator_seed,
        skip=skip
    )
    for j in range(sample_size):
        sample[j, :] = six.next(seq_generator)
    return sample


def i4_bit_hi1(n: int):
    """ Return the position of the high 1 bit base 2 in an integer. """
    # pylint: disable=invalid-name
    i = np.floor(n)
    bit = 0
    while i > 0:
        bit += 1
        i //= 2
    return bit


def i4_bit_lo0(n: int):
    """ Return the position of the low 0 bit base 2 in an integer. """
    # pylint: disable=invalid-name
    bit = 1
    i = np.floor(n)
    while i != 2 * (i // 2):
        bit += 1
        i //= 2
    return bit


DIM_MAX = 40
LOG_MAX = 30
ATMOST = 2 ** LOG_MAX - 1
MAXCOL = i4_bit_hi1(ATMOST)
POLY = [1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109, 103,
        115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171, 213, 191,
        253, 203, 211, 239, 247, 285, 369, 299]


def init_v():
    """Initialise v for the Sobol sequence generator.

    This function provides the array v used by the sobol sequence generator.

    Rather than declaring it as a global variable, it is instanced once inside
    the generator.
    """
    # pylint: disable=invalid-name
    v = np.zeros((DIM_MAX, LOG_MAX))
    v[0:40, 0] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1])

    v[2:40, 1] = np.array([1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1,
                           3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
                           1, 3, 1, 3])

    v[3:40, 2] = np.array([7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
                           5, 3, 3, 1, 7, 5, 1, 3, 3, 7, 5, 1, 1, 5, 7, 7, 5,
                           1, 3, 3])

    v[5:40, 3] = np.array([1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15,
                           5, 3, 15, 7, 9, 13, 9, 1, 11, 7, 5, 15, 1, 15, 11,
                           5, 3, 1, 7, 9])

    v[7:40, 4] = np.array([9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17, 1,
                           25, 29, 3, 31, 11, 5, 23, 27, 19, 21, 5, 1, 17, 13,
                           7, 15, 9, 31, 9])

    v[13:40, 5] = np.array([37, 33, 7, 5, 11, 39, 63, 27, 17, 15, 23, 29, 3,
                            21, 13, 31, 25, 9, 49, 33, 19, 29, 11, 19, 27, 15,
                            25])

    v[19:40, 6] = np.array([13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7,
                            59, 65, 21, 3, 113, 61, 89, 45, 107])

    v[37:40, 7] = np.array([7, 23, 39])

    v[0, 0:MAXCOL] = 1

    return v


def prime_ge(n: int):
    """ Return the smallest prime greater than or equal to n. """
    # pylint: disable=invalid-name
    p = max(np.ceil(n), 2)
    while not is_prime(p):
        p += 1
    return p


def is_prime(n: int) -> bool:
    """ True if n is prime, else False.

    Original version by Corrado Chisari.
    """
    # pylint: disable=invalid-name
    if n != int(n) or n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # All primes >3 are of the form 6n+1 or 6n+5
    # (6n, 6n+2, 6n+4 are 2-divisible, 6n+3 is 3-divisible)
    p = 5
    root = int(np.ceil(np.sqrt(n)))
    while p <= root:
        if n % p == 0 or n % (p + 2) == 0:
            return False
        p += 6
    return True


def i4_sobol2(
        dim_num: int,
        generator_seed: int = 0,
        skip: int = 1
):
    """ Sobol sequence generator.

    This is a modified version of i4_sobol from the sobol_seq package: see
    https://github.com/naught101/sobol_seq/blob/master/sobol_seq).

    The algorithm based on the following original work:

    Antonov, Saleev,
    USSR Computational Mathematics and Mathematical Physics,
    Volume 19, 1980, pages 252 - 256.

    Paul Bratley, Bennett Fox,
    Algorithm 659:
    Implementing Sobol's Quasirandom Sequence Generator,
    ACM Transactions on Mathematical Software,
    Volume 14, Number 1, pages 88-100, 1988.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

    Ilya Sobol,
    USSR Computational Mathematics and Mathematical Physics,
    Volume 16, pages 236-242, 1977.

    Ilya Sobol, Levitan,
    The Production of Points Uniformly Distributed in a Multidimensional Cube
    (in Russian),
    Preprint IPM Akad. Nauk SSSR,
    Number 40, Moscow 1976.

    Notes
    -----
    Needs tidying up. Features a bunch of pylint/flake8 warning suppressions
    that need fixing. Variable "l" in original implementation prefixed "_" to
    silence "bad variable name" warnings.
    """
    # pylint: disable=invalid-name, line-too-long, too-many-locals, too-many-statements, too-many-branches # noqa: E501
    generator_seed_save = -1
    generator_seed = int(np.floor(generator_seed))
    if generator_seed < 0:
        generator_seed = 0
    v = init_v()
    #  Initialize the remaining rows of V.
    for i in range(2, dim_num + 1):
        # The bits of the integer POLY(I) gives the form of polynomial I.
        # Find the degree of polynomial I from binary encoding.
        j = POLY[i - 1]
        m = 0
        j //= 2
        while j > 0:
            j //= 2
            m += 1
        # Expand this bit pattern to separate components of the logical
        # array INCLUD.
        j = POLY[i - 1]
        includ = np.zeros(m)
        for k in range(m, 0, -1):
            j2 = j // 2
            includ[k - 1] = (j != 2 * j2)
            j = j2
        # Calculate the remaining elements of row I as explained in
        # Bratley and Fox, section 2.
        for j in range(m + 1, MAXCOL + 1):
            newv = v[i - 1, j - m - 1]
            _l = 1
            for k in range(1, m + 1):
                _l *= 2
                if includ[k - 1]:
                    newv = np.bitwise_xor(int(newv),
                                          int(_l * v[i - 1, j - k - 1]))
            v[i - 1, j - 1] = newv
    #  Multiply columns of V by appropriate power of 2.
    _l = 1
    for j in range(MAXCOL - 1, 0, -1):
        _l *= 2
        v[0:dim_num, j - 1] = v[0:dim_num, j - 1] * _l
    # RECIPD is 1/(common denominator of the elements in V).
    recipd = 1.0 / (2 * _l)
    lastq = np.zeros(dim_num)
    keep_going = True
    while keep_going:
        _l = 1
        if generator_seed == 0:
            lastq = np.zeros(dim_num)
        elif generator_seed == generator_seed_save + 1:
            _l = i4_bit_lo0(generator_seed)
        elif generator_seed <= generator_seed_save:
            generator_seed_save = 0
            lastq = np.zeros(dim_num)
            for generator_seed_temp in range(int(generator_seed_save), int(generator_seed)):  # noqa: E501
                _l = i4_bit_lo0(generator_seed_temp)
                for i in range(1, dim_num + 1):
                    lastq[i - 1] = np.bitwise_xor(
                        int(lastq[i - 1]), int(v[i - 1, _l - 1]))
            _l = i4_bit_lo0(generator_seed)
        elif generator_seed_save + 1 < generator_seed:
            for generator_seed_temp in range(int(generator_seed_save + 1), int(generator_seed)):  # noqa: E501
                _l = i4_bit_lo0(generator_seed_temp)
                for i in range(1, dim_num + 1):
                    lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]), int(v[i - 1, _l - 1]))  # noqa: E501
            _l = i4_bit_lo0(generator_seed)
        if MAXCOL < _l:
            keep_going = False
        quasi = np.zeros(dim_num)
        for i in range(1, dim_num + 1):
            quasi[i - 1] = lastq[i - 1] * recipd
            lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]), int(v[i - 1, _l - 1]))  # noqa: E501
        # overwite generator_seed_save with current generator_seed
        generator_seed_save = generator_seed
        yield quasi
        generator_seed += (1 + skip)
