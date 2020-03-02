from apricot.core.exceptions import (
    _raise_NotImplemented,
    _raise_NotParsed
)
from apricot.core.models.build.mean_parts import AVAILABLE as _available_mean_functions
from apricot.core.models.build.noise_parts import AVAILABLE as _available_noise_types

def parse_kernel(kernel_type):
    """Parse requested kernel option."""
    return kernel_type

def parse_noise(noise_type):
    """Parse requested noise model."""
    if noise_type is None:
        noise = 0.0
    elif type(noise_type) is str:
        noise = _parse_noise_str(noise_type.lower())
    else:
        noise = _parse_noise_float(noise_type)
    return _parse_noise_internal(noise)

def _parse_noise_str(as_str):
    """Parse noise options that are strings."""
    if as_str == 'zero' or as_str == 'none':
        return 0.0
    if as_str == 'infer':
        return 'infer'
    _raise_NotImplemented('noise function', as_str, _available_noise_types)

def _parse_noise_float(noise_type):
    """Parse noise options that are floating point numbers"""
    try:
        noise = float(noise_type)
    except ValueError:
        raise TypeError("'noise_type' must be either a floating point number, None, 'zero' or 'infer'.")
    return noise

def _parse_noise_internal(noise):
    """Assign noise options to required format"""
    if type(noise) is float:
        noise_option = 'deterministic'
        value = noise
    else:
        noise_option = 'infer'
        value = None
    return noise_option, value

def parse_mean(mean_type):
    """Parse requested mean function."""
    if mean_type is None:
        return 'zero'
    if type(mean_type) is str:
        return _parse_mean_str(mean_type.lower())
    else:
        return _parse_mean_other(mean_type)

def _parse_mean_str(as_str):
    if as_str == 'zero':
        return 'zero'
    elif as_str == 'linear':
        return 'linear'
    else: 
        _raise_NotImplemented('mean function', mean_type, _available_mean_functions)

def _parse_mean_other(mean_type):
    if mean_type == 0:
        return 'zero'
    else:
        _raise_NotParsed('mean function', type(mean_type))

def parse_warping(warping):
    if warping is None:
        return False
    elif type(warping) is str:
        return _parse_warping_string(warping.lower())
    else:
        _raise_NotParsed('warping', type(warping))

def _parse_warping_string(as_str):
    if as_str == 'none':
        return False
    elif as_str == 'linear':
        return 'linear'
    elif as_str == 'sigmoid':
        return 'sigmoid'
