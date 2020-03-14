from apricot.core.models.build.components import StanModelMeanFunction
from apricot.core.models.build.code_snippets import x_to_matrix, mu_to_zeros, x_dot_beta
from apricot.core.exceptions import _raise_NotImplemented


mean_linear = {
    'name': 'linear',
    'data': ['vector[d+1] beta_loc;', 'vector[d+1] beta_scale;'],
    'transformed_data': x_to_matrix,
    'parameters': ['vector[d+1] beta;'],
    'transformed_parameters': x_dot_beta,
    'model': ['beta ~ normal(beta_loc, beta_scale);'],
    'args': [('beta', 'd+1')],
    'to_sample': ['beta'],
    'data_priors': ['beta_loc', 'beta_scale']
}


mean_zero = {
    'name': 'zero',
    'transformed_parameters': mu_to_zeros
}


AVAILABLE = {
    'linear': mean_linear,
    'zero': mean_zero,
}


def find_mean(req_mean_type):
    try:
        mean_function_options = AVAILABLE[req_mean_type]
    except KeyError:
        _raise_NotImplemented('mean function', req_mean_type, AVAILABLE)
    return mean_function_options


def make_mean(mean_type):
    mean_function_options = find_mean(mean_type)
    return StanModelMeanFunction(**mean_function_options)
