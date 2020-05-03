# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
import os
import pickle
from typing import Callable, Any
import six
import pystan  # type: ignore
import apricot
from apricot.core.models import build
from apricot.core.logger import get_logger, PYSTAN_LOGGER_ENABLED


LOGGER = get_logger()


if PYSTAN_LOGGER_ENABLED:
    pystan.api.logger.disabled = 0
else:
    pystan.api.logger.disabled = 1


_ROOTDIR = os.path.dirname(os.path.abspath(apricot.__file__))
_MODEL_CACHE = _ROOTDIR + '/cache/'


KernelPart = build.components.StanModelKernel
MeanPart = build.components.StanModelMeanFunction
NoisePart = build.components.StanModelNoise


def memo(func: Callable[[str], Any]) -> Callable[[str], Any]:
    """ Simple session cache decorator.

    Prevents repeatedly unpickling the same model over and over.
    """
    memory = {}

    def wrapper(string):
        if string not in memory:
            memory[string] = func(string)
        return memory[string]
    return wrapper


def load(
        kernel_part: KernelPart,
        mean_part: MeanPart,
        noise_part: NoisePart,
        warp: bool,
) -> pystan.StanModel:
    filename = get_filename(kernel_part, mean_part, noise_part, warp)
    if os.path.isfile(filename):
        return load_from_pickle(filename)
    return compile_model(kernel_part, mean_part, noise_part, filename)


def get_filename(
        kernel_part: KernelPart,
        mean_part: MeanPart,
        noise_part: NoisePart,
        warp: bool,
) -> str:
    fname = '_'.join([
        kernel_part.filename_component,
        mean_part.filename_component,
        noise_part.filename_component,
    ])
    if warp:
        fname += ('_warped')
    return _MODEL_CACHE + fname + '.pkl'


@memo
def load_from_pickle(filename: str) -> pystan.StanModel:
    """Load a permanently cached pystan model """
    LOGGER.debug('Loading Stan model: %s', filename)
    return pickle.load(open(filename, 'rb'))


def compile_model(
        kernel_part: KernelPart,
        mean_part: MeanPart,
        noise_part: NoisePart,
        filename: str,
) -> pystan.StanModel:
    """ Assemble model code from parts, compile it, and save the pickle. """
    to_cache = prompt_cache()
    model_code = build.assmeble_model_code(kernel_part, mean_part, noise_part)
    compiled_model = pystan.StanModel(model_code=model_code)
    if to_cache:
        with open(filename, 'wb') as destination:
            LOGGER.debug('Saving Stan model: %s', filename)
            pickle.dump(compiled_model, destination)
    return compiled_model


# TODO: add timeout
def prompt_cache(attempts: int = 0) -> bool:
    """ Ask the user if they want to cache the model. """
    attempts += 1
    if attempts > 5:
        raise RuntimeError('Maximum attempts exceeded. Aborted.')
    ans = six.moves.input('Save this model to the cache? [y/n/c]: ').lower()
    if ans == 'c':
        raise RuntimeError('Cancelled by user.')
    if ans == 'y':
        return True
    if ans == 'n':
        return False
    print("Answer either (y)es, (n)o, or (c)ancel.")
    return prompt_cache(attempts=attempts)
