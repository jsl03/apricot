# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from apricot.core import utils
from apricot.core.models.build import components
from apricot.core.models.build import core_parts


def assmeble_model_code(
        kernel_part: components.StanModelKernel,
        mean_part: components.StanModelMeanFunction,
        noise_part: components.StanModelNoise,
) -> str:
    """ Assemble the pyStan model code.

    Parameters
    ----------
    kernel_part : components.StanModelKernel
        Kernel related options for the model.
    mean_part : components.StanModelMeanFunction
        Mean function related options for the model.
    noise_part : components.StandModelNoise
        Noise related options for the model.

    Returns
    -------
    model_code : str
        pyStan model code, as a string.
    """
    core_part = core_parts._get_core()
    return _fuse_code_blocks(core_part, kernel_part, mean_part, noise_part)


def _fuse_code_blocks(
        core: components.StanModelPart,
        kernel: components.StanModelKernel,
        mean: components.StanModelMeanFunction,
        noise: components.StanModelNoise,
) -> str:
    """ Fuse the code blocks together.

    Parameters
    ----------
    core_part : components.StanModelPart
        Generic ('core') model components.
    kernel_part : components.StanModelKernel
        Kernel related options for the model.
    mean_part : components.StanModelMeanFunction
        Mean function related options for the model.
    noise_part : components.StandModelNoise
        Noise related options for the model.

    Returns
    -------
    model_code : str
        pyStan model code, as a string.
    """
    names = (
        'functions',
        'data',
        'transformed data',
        'parameters',
        'transformed parameters',
        'model',
    )
    zipped = zip(core, kernel, mean, noise)

    # join the code in each block with newlines
    model_code = [utils.join_strings([c, k, m, n]) for c, k, m, n in zipped]

    # ugly way of appending spaces to each newline so the model code is easier
    # to read if printed to the console
    formatted_blocks = []
    for block in model_code:
        formatted_blocks.append(block.replace('\n', '\n  '))

    model_blocks = (_fuse(n, b) for n, b in zip(names, formatted_blocks))

    # join blocks together with newlines and we're done
    return '\n'.join(model_blocks)


def _fuse(name: str, block: str) -> str:
    return '{0} {{\n  {1}\n}}'.format(name, block)
