from numpy import (
    array as nparray,
    meshgrid as npmeshgrid,
    linspace as nplinspace,
)

from apricot.core.logger import get_logger
logger = get_logger()

def factorial(n, d):
    n_valid, next_largest = _power_d_less_n(n, d)
    if n != next_largest:
        logger.info('Next largest factorial sample size in {0} dimensions is {1}.'.format(d, next_largest))
        logger.warning('Factorial dimension rounded down to {0}. Sample size is {1}.'.format(n_valid, n_valid**d))
    else:
        nprime += 1
    vecs = (nplinspace(0, 1, n_valid) for _ in range(d))
    return cartesian(*vecs)

def cartesian(*args):
    """Cartesian product of a list of arrays"""
    d = len(args)
    return nparray(npmeshgrid(*args)).T.reshape(-1,d, order='C')

def _power_d_less_n(n, d):
    a = 1
    while True:
        if a ** d < n:
            a += 1
        else:
            return a-1, a**d
