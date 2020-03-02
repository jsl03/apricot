"""
This module is a stripped down and slightly modified version of the
"sobol_seq" package.

Please see the original code, which is available at:
https://github.com/naught101/sobol_seq

sobol_seq is in turn based on original contributions as follows:

Original FORTRAN77 version of i4_sobol by Bennett Fox.
MATLAB version by John Burkardt.
PYTHON version by Corrado Chisari
Original Python version of is_prime by Corrado Chisari
Original MATLAB versions of other functions by John Burkardt.
PYTHON versions by Corrado Chisari
Original code is available at http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
"""

from __future__ import division
import numpy as np
import six

def sobol_scatter(n, d, seed=1, skip=0):
    """ Generate n length d quasi-random vectors using the Sobol sequence, with
    additive uniform randomization.

    Implements i4sobol_generate from the sobol_seq package but without the
    ability to change the dimension of the sequence after it has been
    initialised, and with randomization applied to the generated points.

    Parameters
    ----------
    n : int
        The number of random vectors to retrieve.
    d : int
        The dimension of the random vectors

    Returns
    -------
    samples : ndarray
        (n,d) array consisting of n quasi-random vectors in d dimensions.

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
    r = np.empty((n, d))
    seq_generator = i4_sobol2(d, seed=seed, skip=skip)
    for j in range(n):
        r[j, :] = six.next(seq_generator)

    r += np.random.random(size=r.shape)
    while( (r<0).any() | (r>1).any()):
        r[r>1] -= 1
        r[r<0] += 1

    return r

def sobol(n, d, seed=1, skip=0):
    """ Generate n length d quasi-random vectors from the Sobol sequence.

    Generate n length d quasi-random vectors using the Sobol sequence [1]_.

    Implements i4sobol_generate from the sobol_seq package but without the
    ability to change the dimension of the sequence after it has been
    initialised.

    Parameters
    ----------
    n : int
        The number of random vectors to retrieve.
    d : int
        The dimension of the random vectors

    Returns
    -------
    samples : ndarray
        (n,d) array consisting of n quasi-random vectors in d dimensions.

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
    r = np.empty((n, d))
    seq_generator = i4_sobol2(d, seed=seed, skip=skip)
    for j in range(n):
        r[j, :] = six.next(seq_generator)

    return r

def i4_bit_hi1(n):
    """Return the position of the high 1 bit base 2 in an integer. """
    i = np.floor(n)
    bit = 0
    while i > 0:
        bit += 1
        i //= 2
    return bit

def i4_bit_lo0(n):
    """Return the position of the low 0 bit base 2 in an integer."""
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
    v = np.zeros((DIM_MAX, LOG_MAX))
    v[0:40, 0] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1])

    v[2:40, 1] = np.array([1, 3, 1, 3, 1, 3, 3, 1, 3, 1,  3, 1, 3, 1, 1, 3, 1,
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
                           7,  15, 9,  31, 9])

    v[13:40, 5] = np.array([37, 33, 7,  5, 11, 39, 63, 27, 17, 15, 23, 29, 3,
                            21, 13, 31, 25, 9, 49, 33, 19, 29, 11, 19, 27, 15,
                            25])

    v[19:40, 6] = np.array([13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7,
                            59, 65, 21, 3, 113, 61, 89, 45, 107])


    v[37:40, 7] = np.array([7, 23, 39])

    
    v[0, 0:MAXCOL] = 1

    return v

def prime_ge(n):
    """Return the smallest prime greater than or equal to N."""
    p = max(np.ceil(n), 2)
    while not is_prime(p):
        p += 1
    return p

def is_prime(n):
    """True if n is prime, else False.

    Original version by Corrado Chisari.
    """
    if n != int(n) or n < 2:
        return False
    if n == 2 or n == 3:
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

def i4_sobol2(dim_num, seed=0, skip=1):
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
    The Production of Points Uniformly Distributed in a Multidimensional Cube (in Russian),
    Preprint IPM Akad. Nauk SSSR,
    Number 40, Moscow 1976.
    """
    seed_save = -1
    seed = int(np.floor(seed))
    if seed < 0:
        seed = 0

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
            l = 1
            for k in range(1, m + 1):
                l *= 2
                if includ[k - 1]:
                    newv = np.bitwise_xor(int(newv),
                                          int(l * v[i - 1, j - k - 1]))
            v[i - 1, j - 1] = newv

    #  Multiply columns of V by appropriate power of 2.
    l = 1
    for j in range(MAXCOL - 1, 0, -1):
        l *= 2
        v[0:dim_num, j - 1] = v[0:dim_num, j - 1] * l

    # RECIPD is 1/(common denominator of the elements in V).
    recipd = 1.0 / (2 * l)
    lastq = np.zeros(dim_num)
    keep_going = True
    while keep_going:
        l = 1
        if seed == 0:
            lastq = np.zeros(dim_num)
        elif seed == seed_save + 1:
            l = i4_bit_lo0(seed)
        elif seed <= seed_save:
            seed_save = 0
            lastq = np.zeros(dim_num)
            for seed_temp in range(int(seed_save), int(seed)):
                l = i4_bit_lo0(seed_temp)
                for i in range(1, dim_num + 1):
                    lastq[i - 1] = np.bitwise_xor(
                        int(lastq[i - 1]), int(v[i - 1, l - 1]))
            l = i4_bit_lo0(seed)
        elif seed_save + 1 < seed:
            for seed_temp in range(int(seed_save + 1), int(seed)):
                l = i4_bit_lo0(seed_temp)
                for i in range(1, dim_num + 1):
                    lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]),
                                                  int(v[i - 1, l - 1]))
            l = i4_bit_lo0(seed)
        if MAXCOL < l:
            keep_going = False
        quasi = np.zeros(dim_num)
        for i in range(1, dim_num + 1):
            quasi[i - 1] = lastq[i - 1] * recipd
            lastq[i - 1] = np.bitwise_xor(int(lastq[i - 1]), int(v[i - 1, l - 1]))
        
        # overwite seed_save with current seed 
        seed_save = seed
        yield quasi
        seed += (1 + skip)
