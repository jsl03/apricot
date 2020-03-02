from apricot.core.utils import join_strings
from apricot.core.models.build.core_parts import _get_core

def assmeble_model_code(kernel_part, mean_part, noise_part):
    core_part = _get_core()
    return _fuse_code_blocks(core_part, kernel_part, mean_part, noise_part)

def _fuse_code_blocks(core, kernel, mean, noise):
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
    model_code = [join_strings([c, k, m, n]) for c, k, m, n in zipped]

    # ugly way of appending spaces to each newline so the model code is easier
    # to read if printed to the console
    formatted_blocks = []
    for block in model_code:
        formatted_blocks.append(block.replace('\n', '\n  '))

    # fuse the blocks together with the appropriate names
    fuse = lambda name, block : '{0} {{\n  {1}\n}}'.format(name, block)
    model_blocks = (fuse(n, b) for n, b in zip(names, formatted_blocks))

    # join blocks together with newlines and we're done
    return '\n'.join(model_blocks)
