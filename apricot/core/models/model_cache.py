# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
import os
import pickle
import six

import apricot
import pystan

from apricot.core.models import build

# disable pystan's logger as we will run our own diagnostics
pystan.api.logger.disabled = 1

_ROOTDIR = os.path.dirname(os.path.abspath(apricot.__file__))
_MODEL_CACHE = _ROOTDIR + '/cache/'

# TODO: how does mypy check for inheritance?
Model_Part_Type = build.components.StanModelPart


def memo(func: callable) -> callable:
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
        kernel_part: Model_Part_Type,
        mean_part: Model_Part_Type,
        noise_part: Model_Part_Type,
        warp: bool,
) -> pystan.StanModel:
    filename = get_filename(kernel_part, mean_part, noise_part, warp)
    if os.path.isfile(filename):
        return load_from_pickle(filename)
    else:
        return compile_model(kernel_part, mean_part, noise_part, filename)


def get_filename(
        kernel_part: Model_Part_Type,
        mean_part: Model_Part_Type,
        noise_part: Model_Part_Type,
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
    return pickle.load(open(filename, 'rb'))


def compile_model(
        kernel_part: Model_Part_Type,
        mean_part: Model_Part_Type,
        noise_part: Model_Part_Type,
        filename: str,
) -> pystan.StanModel:
    """ Assemble model code from parts, compile it, and save the pickle. """
    to_cache = prompt_cache()
    model_code = build.assmeble_model_code(kernel_part, mean_part, noise_part)
    compiled_model = pystan.StanModel(model_code = model_code)
    if to_cache:
        with open(filename, 'wb') as destination:
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
    else:
        print("Answer either (y)es, (n)o, or (c)ancel.")
        return prompt_cache(attempts=attempts)
