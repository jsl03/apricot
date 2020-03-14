from apricot.core.models.build.components import StanModelNoise
from apricot.core.exceptions import _raise_NotImplemented


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


def find_noise(req_noise_type):
    try:
        noise_options = AVAILABLE[req_noise_type[0]]
    except KeyError:
        _raise_NotImplemented('noise type', req_noise_type, AVAILABLE)
    return noise_options


def make_noise(noise_type):
    noise_options = find_noise(noise_type)
    return StanModelNoise(**noise_options)
