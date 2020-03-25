import typing
from apricot.core import exceptions
from apricot.core.models.build import mean_parts
from apricot.core.models.build import noise_parts


# TODO: this is broken: doesnt parse anything. Need to fix so apricot.Emulator
# can use it, too.
def parse_kernel(kernel_type: str):
    """Parse requested kernel option."""
    return kernel_type


def parse_noise(noise_type: typing.Optional[typing.Union[float, str]]):
    """ Parse requested noise option.

    Parameters
    ----------
    noise_type : {None, str, float}
        * None : zero noise.
        * 'zero' : zero noise.
        * 'infer' : infer the standard deviation of Gaussian white noise as a
        hyperparameter.
        * float : Gaussian white noise of standard deviation equivalent to
        noise_type.

    Returns
    -------
    noise_option : {'deterministic', 'infer'}
        Either 'deterministic' or 'infer'.
    noise_value : {None, float}
        Either a floating point number or None.
    """
    if noise_type is None:
        noise = 0.0
    elif type(noise_type) is str:
        noise = _parse_noise_str(noise_type.lower())
    else:
        noise = _parse_noise_float(noise_type)
    return _parse_noise_internal(noise)


def _parse_noise_str(as_str: str):
    """ Parse noise options that are strings. """
    if as_str == 'zero' or as_str == 'none':
        return 0.0
    if as_str == 'infer':
        return 'infer'
    exceptions._raise_NotImplemented(
        'noise function',
        as_str,
        noise_parts.AVAILABLE
    )


def _parse_noise_float(noise_type: float):
    """ Parse noise options that are floating point numbers. """
    try:
        noise = float(noise_type)
    except ValueError:
        raise TypeError("'noise_type' must be either a floating point number, None, 'zero' or 'infer'.")
    return noise


def _parse_noise_internal(noise: typing.Union[str, float]):
    """ Assign noise options to required format. """
    if type(noise) is float:
        noise_option = 'deterministic'
        value = noise
    else:
        noise_option = 'infer'
        value = None
    return noise_option, value


def parse_mean(mean_type: typing.Optional[typing.Union[str, float]]):
    """ Parse requested mean function option.

    Parameters
    ----------
    mean_type : {None, str, int}
        * None : zero mean
        * 'zero' : zero mean
        * 'linear' : linear mean
        * 0 : zero mean

    Returns
    -------
    parsed_mean_type : str
        Either 'zero' or 'linear'.

    Notes
    -----
    'linear' will compile but no predictive model for a linear mean
    currently exists inside apricot/src!
    """
    if mean_type is None:
        return 'zero'
    if type(mean_type) is str:
        return _parse_mean_str(mean_type.lower())
    else:
        return _parse_mean_other(mean_type)


def _parse_mean_str(as_str: str):
    if as_str == 'zero':
        return 'zero'
    elif as_str == 'linear':
        return 'linear'
    else:
        exceptions._raise_NotImplemented(
            'mean function',
            as_str,
            mean_parts.AVAILABLE
        )


# realistically, mean_type can only be 0 or we raise an exception
def _parse_mean_other(mean_type: int):
    if mean_type == 0:
        return 'zero'
    else:
        exceptions._raise_NotParsed('mean function', type(mean_type))


def parse_warping(warping: typing.Optional[typing.Union[bool, str]]):
    """ Parse warping option.

    Parameters
    ----------
    warping : {None, str, bool}
        * None : no warping.
        * False : no warping.
        * 'linear' : approximately linear warping.
        * 'sigmoid' : approximately sigmoidal warping.

    Returns
    -------
    warping_type : {bool, str}
        One of {False, 'linear', 'sigmoid'}

    """
    if warping is None:
        return False
    elif type(warping) is str:
        return _parse_warping_string(warping.lower())
    else:
        exceptions._raise_NotParsed('warping', type(warping))


def _parse_warping_string(as_str: str):
    if as_str == 'none':
        return False
    elif as_str == 'linear':
        return 'linear'
    elif as_str == 'sigmoid':
        return 'sigmoid'
