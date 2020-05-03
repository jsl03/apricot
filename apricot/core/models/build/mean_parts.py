# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import (
    Dict,
    Any,
    Mapping,
    Union,
    Iterable
)
from apricot.core.models.build.code_snippets import (
    X_TO_MATRIX,
    MU_TO_ZEROS,
    X_DOT_BETA,
)
from apricot.core.models.build.components import StanModelMeanFunction


MEAN_LINEAR: Mapping[str, Union[str, Iterable[Any]]] = {
    'name': 'linear',
    'data': ['vector[d+1] beta_loc;', 'vector[d+1] beta_scale;'],
    'transformed_data': X_TO_MATRIX,
    'parameters': ['vector[d+1] beta;'],
    'transformed_parameters': X_DOT_BETA,
    'model': ['beta ~ normal(beta_loc, beta_scale);'],
    'args': [('beta', 'd+1')],
    'to_sample': ['beta'],
    'data_priors': ['beta_loc', 'beta_scale']
}


MEAN_ZERO: Mapping[str, Union[str, Iterable[str]]] = {
    'name': 'zero',
    'transformed_parameters': MU_TO_ZEROS
}


AVAILABLE: Mapping[str, Mapping[str, Union[str, Iterable[str]]]] = {
    'linear': MEAN_LINEAR,
    'zero': MEAN_ZERO,
}


def make_mean(mean_type: str) -> StanModelMeanFunction:
    return StanModelMeanFunction(**AVAILABLE[mean_type])
