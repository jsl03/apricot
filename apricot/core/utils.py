import os
import itertools
import numpy as np

from functools import wraps

import apricot

_ROOTDIR = os.path.dirname(os.path.abspath(apricot.__file__))
_MODEL_CACHE = _ROOTDIR + '/cache/'

def mad(arr, axis=None):
    """Median absolute deviation"""
    deviation = np.abs(arr - np.mean(arr, axis=axis))
    return np.median(deviation, axis=axis)

def maybe(func):
    """ Decorator for functions which fail if any arguments are None.

    If any arguments passed to a function wrapped with maybe are None,
    execution of the wrapped function will be bypassed and None will be
    returned instead.

    Parameters
    ----------
    func : callable
        The function to be wrapped

    Returns
    -------
    wrapped : callable
        The provided function, wrapped with maybe
    """
    @wraps(func)
    def wrapper(*args):
        if any(x is None for x in args):
            return None
        else:
            return func(*args)
    return wrapper

@maybe
def _force_f_array(arr):
    """Force numpy array to be F contiguous."""
    if arr.flags['F_CONTIGUOUS']:
        return arr
    else:
        return np.asfortranarray(arr)

@maybe
def _atleast2d_fview(arr):
    """Force an array to be at least 2D and F contiguous."""
    if arr.ndim == 1:
        return arr.reshape(-1, 1, order='F')
    else:
        return _force_f_array(arr)

def random_seed():
    """Generate pyStan Compatible Random Seed

    Note
    ----
    Until scipy.optimize accepts a random seed, some apricot methods cannot be
    made deterministic since the results of preliminary optimisations (e.g. to
    construct an empirically derived prior distribution) cannot be tightly
    controlled.
    """
    return np.random.randint(np.iinfo(np.int32).max)

def set_seed(seed):
    if seed is None:
        np.random.seed(random_seed())
    else:
        np.random.seed(seed)

def is_string(x):
    """Return True if x is a string of finite length"""
    if x is None:
        return False
    if type(x) is not str:
        return False
    if len(x) == 0:
        return False
    return True

def join_strings(seq):
    """Join all of the elements of seq that are not None"""
    return '\n'.join([elem for elem in seq if is_string(elem)])

def to_list(x):
    """If x is not a list, make it into a list"""
    if type(x) is list:
        return x
    else:
        return [x]

def join_lines(x):
    """If x is a list, apply _join_strings. If not, do nothing."""
    if x is None:
        return None
    if type(x) is list:
        return join_strings(x)
    else:
        return x

def flatten(list2d):
    """Flatten 2d iterable of iterables into 1d list"""
    return list(itertools.chain.from_iterable(list2d))

def inspect_cache():
    """Inspect the contents of the model cache

    Lists all .pkl files in the default cache location.
    """
    cached_models = []
    for path, dirs, files in os.walk(_MODEL_CACHE):
        for file in files:
            if file.endswith(".pkl"):
                cached_models.append(path + file)
    return cached_models

def clear_model_cache():
    """Clears the model cache

    Removes all .pkl files from the default cache location.
    """
    for path, dirs, files in os.walk(_MODEL_CACHE):
        for file in files:
            if file.endswith(".pkl"):
                print('Removed file: {0}'.format(path + file))
                os.remove(path + file)

