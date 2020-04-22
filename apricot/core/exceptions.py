# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------


def _raise_NotImplemented(
        model_part_identifier: str,
        name: str,
        available: dict
) -> None:
    """ Standard error format for requested options that do not exist.

    Raises
    ------
    NotImplementedError
    """

    raise NotImplementedError(
        'Unrecognised {0} "{1}": must be one of {2}'.format(
            model_part_identifier,
            name,
            available.keys()
        )
    ) from None


def _raise_NotParsed(name: str, _type: type) -> None:
    """ Standard error format for arguments that could not be parsed.

    Raises
    ------
    TypeError
    """
    raise TypeError(
        "Could not parse {n} with type {t}.".format(n=name, t=_type)
    ) from None


class ShapeError(Exception):
    """ Exception class for when inputs of the wrong shape are provided to
    a model or function expecting a specific number of rows and/or columns.
    """

    def __init__(
            self,
            identifier: str,
            d0_required: int,
            d1_required: int,
            provided: (int, int),
    ) -> None:

        self.info = {
            'arr': identifier,
            'r0': d0_required,
            'r1': d1_required,
            'p': provided,
        }

    def __str__(self) -> str:
        return "Received array '{arr}' of shape {p}: "
        "shape of ({r0}, {r1}) required.".format(**self.info)


class MissingParameterError(Exception):
    """Exception class for when a model tries to access a hyperparameter
    not provided by the hyperparameter dictionary."""

    def __init__(self, name: str) -> None:
        self.message = ("Model tried to access unavailable hyperparameter "
                        "{0}".format(name))

    def __str__(self) -> str:
        return self.message
