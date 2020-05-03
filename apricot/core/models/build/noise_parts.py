# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from apricot.core.models.build.components import StanModelNoise


noise_infer = {
    'name': 'infer',
    'data': 'real<lower=0> xi_scale;',
    'parameters': 'real<lower=0> xi;',
    'model': 'xi ~ normal(0, xi_scale);',
    'args': [('xi', 1)],
    'to_sample': ['xi'],
    'data_priors': ['xi_scale']
}


noise_deterministic = {
    'name': 'deterministic',
    'data': 'real<lower=0> xi;',
    'args': ('xi', 1),
    'data_priors': ['xi'],
}


AVAILABLE = {
    'infer': noise_infer,
    'deterministic': noise_deterministic,
}


def make_noise(noise_type):
    return StanModelNoise(**AVAILABLE[noise_type])
