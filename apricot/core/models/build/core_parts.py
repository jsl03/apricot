from apricot.core.models.build.components import StanModelPart

_CORE = {
    'name': None,
    'data': [
        'int<lower=1> n;',
        'int<lower=1> d;',
        'vector[d] x[n];',
        'vector[n] y;',
        'real<lower=0> jitter;',
    ],
    'transformed_parameters': [
        'matrix[n,n] L;',
        'vector[n] mu;'
    ],
    'model': 'y ~ multi_normal_cholesky(mu, L);'
}


def _get_core():
    return StanModelPart(**_CORE)
