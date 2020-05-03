# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import Union, Optional, Tuple
from apricot.core import exceptions
from apricot.core.logger import get_logger
from apricot.core.models.build import mean_parts
from apricot.core.models.build import noise_parts
from apricot.core.models.build import kernel_parts


LOGGER = get_logger()


def parse_kernel(kernel_type: str):
    """Parse requested kernel option."""
    if kernel_type in kernel_parts.AVAILABLE:
        return kernel_type
    msg = (
        "kernel must be one of {0}"
        .format(kernel_parts.AVAILABLE)
    )
    raise ValueError(msg)


def parse_noise(
        noise_type: Optional[Union[str, float]]
) -> Tuple[str, Optional[float]]:
    """ Parse requested noise option. 

    Parameters
    ----------
    noise_type : {None, str, float}
        * None : zero noise.
        * 'zero' : zero noise.
        * 'infer' : infer the standard deviation of additivie Gaussian (white)
            noise as a model hyperparameter.
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
        return parse_noise_internal(0)

    elif isinstance(noise_type, str):
        return parse_noise_internal(parse_noise_str(noise_type.lower()))

    else:
        return parse_noise_internal(parse_noise_float(noise_type))


def parse_noise_str(as_str: str) -> Union[float, str]:
    """ Parse noise options that are strings.

    Parameters
    ----------
    as_str: str
        String instance describing a noise type option:
        * 'zero': zero noise; equivalent to passing noise = 0.
        * 'none': treated equivalently to 'zero' (see above).
        * 'infer': treat the standard deviation of additive Gaussian noise as a
             model hyperparameter and infer it as part of the model.

    Returns
    -------
    noise: {str, float}
        Either a floating point number representing the deterministic standard
        deviation of additive Gaussian noise; or "infer", indicating the
        standard deviation of additive Gaussian noise should be treated as a
        model hyperparameter.
    """
    if as_str in ['zero', 'none']:
        return 0.0
    if as_str == 'infer':
        return 'infer'
    msg = (
        "noise model must be one of {0}"
        .format(noise_parts.AVAILABLE)
    )
    raise ValueError(msg)


def parse_noise_float(noise_type: float) -> float:
    """ Parse noise options that can be treated as floating point numbers.

    Attempts to cast noise_type as a float. This is necessary as Stan
    will not implcitly convert (for example) an integer to a floating point
    number.

    Parameters
    ----------
    noise_type: {float, int}
        User supplied noise type option. Either the value of additive Gaussian
        noise represented as a string or a floating point number.

    Returns
    -------
    noise: float
        User supplied noise option, cast as a floating point number.

    Raises
    ------
    TypeError
        If noise_type cannot be cast as a float (noise options matching
        compatible string options are matched before here).
    """
    try:
        noise = float(noise_type)
    except ValueError:
        raise TypeError(
            "'noise_type' must be either a floating point number, "
            "None, 'zero' or 'infer'."
        )
    return noise


def parse_noise_internal(noise: Union[str, float]
) -> Tuple[str, Optional[float]]:
    """ Assign noise options to required format.

    Really just packages noise_option and its value (if present) into a tuple.

    Parameters
    ----------
    noise: {str, float}
        Noise option. If noise is a float, it is assumed to represent the
        (deterministic) standard deviation of additive Gaussian noise to be
        applied to the model. If noise is a string, it must match "infer", in
        which case the standard deviation of additive Gaussian noise will be
        treated as a hyperparameter and estimated from data.

    Returns
    -------
    noise_option: str
        Either "deterministic" or "infer".
    value: {float, None}
        If noise_option = "deterministic", contains the standard deviation of
        the additive Gaussian noise to include in the model. If noise_option =
        "infer", value is None.
    """
    if isinstance(noise, float):
        return 'deterministic', noise
    return 'infer', None


def parse_mean(mean_type: Optional[Union[str, int]]) -> str:
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
    if isinstance(mean_type, str):
        return parse_mean_str(mean_type.lower())
    return parse_mean_other(mean_type)


def parse_mean_str(as_str: str) -> str:
    if as_str == 'zero':
        return 'zero'
    if as_str == 'linear':
        LOGGER.warning('Linear mean not yet supported by internal GP!')
        return 'linear'
    msg = (
        "mean function must be one of {0}"
        .format(mean_parts.AVAILABLE)
    )
    raise ValueError(msg)


def parse_mean_other(mean_type: int) -> str:
    if mean_type == 0:
        return 'zero'
    msg = (
        "mean function must be one of {0}"
        .format(mean_parts.AVAILABLE)
    )
    raise ValueError(msg)


def parse_warping(warping: Optional[str]) -> Union[bool, str]:
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
    if isinstance(warping, str):
        return parse_warping_string(warping.lower())
    msg = 'Could not parse warping option with type {0}.'.format(type(warping))
    raise TypeError(msg)


def parse_warping_string(as_str: str) -> Union[bool, str]:
    if as_str == 'none':
        return False
    if as_str == 'linear':
        return 'linear'
    if as_str == 'sigmoid':
        return 'sigmoid'
    msg = 'Uncrecognised warping option "{0}": must be one of {1}'.format(
        as_str, [None, 'none', 'linear', 'sigmoid']
    )
    raise ValueError(msg)
