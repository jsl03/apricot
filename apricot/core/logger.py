"""

DOCSTRING

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
import logging

# imported by core.models.model_cache: global variable is here to keep all
# logging utilities in the same place
PYSTAN_LOGGER_ENABLED = False


def get_logger():
    """ Utility to initialise/ return the apricot logging utility. """
    logger = logging.getLogger('apricot')
    logger.propagate = 0
    fmt = logging.Formatter(
        '%(levelname)s: %(name)s: %(message)s')
#       '%(levelname)s: %(name)s: %(module)s:%(funcName)s - %(message)s')
    if not logger.handlers:
        chf = logging.StreamHandler()
        chf.setFormatter(fmt)
        logger.addHandler(chf)
    logger.setLevel(logging.INFO)
    return logger
