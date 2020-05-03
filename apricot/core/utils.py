# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import Iterable, List, Any, Optional, Union, Callable
import os
import itertools
from functools import wraps
import numpy as np  # type: ignore
import apricot
from apricot.core.logger import get_logger


LOGGER = get_logger()
_ROOTDIR = os.path.dirname(os.path.abspath(apricot.__file__))
_MODEL_CACHE = _ROOTDIR + '/cache/'


# TODO: belongs in .math?
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


def maybe(func: Callable) -> Callable:
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
        return func(*args)
    return wrapper


@maybe
def force_f_array(arr: np.ndarray) -> np.ndarray:
    """ Force provided numpy array to be F contiguous. """
    if arr.flags['F_CONTIGUOUS']:
        return arr
    return np.asfortranarray(arr)


@maybe
def _atleast2d_fview(arr: np.ndarray) -> np.ndarray:
    """ Force provided numpy array to be at least 2D and F contiguous. """
    if arr.ndim == 1:
        return arr.reshape(-1, 1, order='F')
    return force_f_array(arr)


def random_seed(seed: Optional[int] = None) -> int:
    """ Generate pyStan Compatible Random Seed. """
    if seed:
        np.random.seed(seed)
    new_seed = np.random.randint(np.iinfo(np.int32).max)
    LOGGER.debug('generated random seed: %s', new_seed)
    return new_seed


def set_seed(seed: Optional[int]) -> None:
    """ Seed numpy's random state. """
    if seed is None:
        seed = random_seed()
        LOGGER.debug('random seed set: %s', seed)
        np.random.seed(seed)
    else:
        LOGGER.debug('random seed set: %s', seed)
        np.random.seed(seed)


def is_string(thing: Optional[Any]) -> bool:
    """Return True if thing is a string of finite length"""
    if thing is None:
        return False
    if isinstance(thing, str):
        # make sure s is of finite length
        if len(thing) == 0:
            return False
        return True
    return False


def join_strings(sequence: Iterable[Any]) -> str:
    """ Join all of the elements of seq that are not None with newlines. """
    elements = [elem for elem in sequence if is_string(elem)]
    if len(elements) > 0:
        return '\n'.join(elements)
    return ''


def to_list(thing: Any) -> List[Any]:
    """ If thing is not already a list, make it into a list. """
    if isinstance(thing, list):
        return thing
    return [thing]


def join_lines(thing: Any) -> Optional[Union[Any, str]]:
    """ If thing is a list, apply _join_strings and return the result. If not,
    return the thing unchanged. """
    if thing is None:
        return None
    if isinstance(thing, list):
        return join_strings(thing)
    return thing


def flatten(list2d: Iterable[Iterable[Any]]) -> List[Any]:
    """ Flatten 2d iterable of iterables into 1d list """
    return list(itertools.chain.from_iterable(list2d))


def inspect_cache() -> List[str]:
    """Inspect the contents of the model cache.

    Lists all .pkl files in the default cache location.
    """
    cached_models = []
    for path, _, files in os.walk(_MODEL_CACHE):
        for file in files:
            if file.endswith(".pkl"):
                cached_models.append(path + file)
    return cached_models


def clear_model_cache() -> None:
    """Clears the model cache

    Removes all .pkl files from the default cache location.
    """
    for path, _, files in os.walk(_MODEL_CACHE):
        for file in files:
            if file.endswith(".pkl"):
                print('Removed file: {0}'.format(path + file))
                os.remove(path + file)


def show_keys(dictionary: dict) -> str:
    """ Format a string of the all of the keys of dictionary, each key on a
    new line."""
    return '\n'+'\n'.join(dictionary.keys())
