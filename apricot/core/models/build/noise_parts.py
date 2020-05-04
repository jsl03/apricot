# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from apricot.core.models.build.components import StanModelNoise


noise_infer = {
    'name': 'infer',
    'data': 'real<lower=0> sigma_scale;',
    'parameters': 'real<lower=0> sigma;',
    'model': 'sigma ~ normal(0, sigma_scale);',
    'args': [('sigma', 1)],
    'to_sample': ['sigma'],
    'data_priors': ['sigma_scale']
}


noise_deterministic = {
    'name': 'deterministic',
    'data': 'real<lower=0> sigma;',
    'args': ('sigma', 1),
    'data_priors': ['sigma'],
}


AVAILABLE = {
    'infer': noise_infer,
    'deterministic': noise_deterministic,
}


def make_noise(noise_type):
    """ Make the Stan model part for the requested noise function. """
    return StanModelNoise(**AVAILABLE[noise_type])
