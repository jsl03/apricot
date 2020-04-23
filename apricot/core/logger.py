# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
import logging
import json
import numpy as np

# imported by core.models.model_cache: global variable is here to keep all
# logging utilities in the same place
PYSTAN_LOGGER_ENABLED = False


def get_logger():
    logger = logging.getLogger('apricot')
    logger.propagate = 0
    fmt = logging.Formatter(
        '%(levelname)s: %(name)s: %(message)s')
#       '%(levelname)s: %(name)s: %(module)s:%(funcName)s - %(message)s')
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger


def log_options(opts):
    return json.dumps(opts, sort_keys=False, indent=4, cls=NumpyEncoder)


# TODO need a way of pretty printing numpy arrays
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # return self.numpy_to_string(obj)
            return '<numpy array>'
        return json.JSONEncoder.default(self, obj)
