import copy
from apricot.core.models.build.components import StanModelKernel
from apricot.core.models.build.code_snippets import (
    L_cov_expq_xi,
    L_cov_m52_xi,
    L_cov_m32_xi,
    L_cov_rq_xi,
    input_warping
)
from apricot.core.exceptions import _raise_NotImplemented

def find_kernel(name, warping):
    try:
        kernel_options = AVAILABLE[name]
    except KeyError:
        _raise_NotImplemented('kernel', name, AVAILABLE)
    if warping:
        return _apply_warping(kernel_options)
    else:
        return _apply(kernel_options)

def make_kernel(kernel_type, warping):
    kernel_options = find_kernel(kernel_type, warping)
    return StanModelKernel(**kernel_options)

GP_DATA_GENERIC = [
    'real<lower=0> amp_loc;',
    'real<lower=0> amp_scale;',
    'vector<lower=0>[d] ls_alpha;',
    'vector<lower=0>[d] ls_beta;',
]

GP_PARAMETERS_GENERIC = [
    'real<lower=0> amp;',
    'vector<lower=0>[d] ls;',
]

GP_MODEL_GENERIC = [
    'amp ~ student_t(3, amp_loc, amp_scale);',
    'ls ~ inv_gamma(ls_alpha, ls_beta);',
]

GP_ARGS_GENERIC = [
    ('amp', 1),
    ('ls', 'd')
]

GP_DATA_PRIORS_GENERIC = [
    'amp_loc',
    'amp_scale',
    'ls_alpha',
    'ls_beta',
]

kernel_expq = {
    'name' : 'expq',
    'functions' : L_cov_expq_xi,
    'data' : GP_DATA_GENERIC, 
    'transformed_data' : None,
    'parameters' : GP_PARAMETERS_GENERIC,
    'kernel_signature' : 'L_cov_expq_xi(__x__, amp, ls, xi, jitter, n)',
    'model' : GP_MODEL_GENERIC,
    'args' : GP_ARGS_GENERIC,
    'to_sample' : ['amp', 'ls'],
    'data_priors' : GP_DATA_PRIORS_GENERIC,
}

kernel_expq_flat = {
    'name' : 'expq_flat',
    'functions' : L_cov_expq_xi,
    'data' : [],
    'transformed_data' : None,
    'parameters' : GP_PARAMETERS_GENERIC,
    'kernel_signature' : 'L_cov_expq_xi(__x__, amp, ls, xi, jitter, n)',
    'model' : [],
    'args' : GP_ARGS_GENERIC,
    'to_sample' : ['amp', 'ls'],
    'data_priors' : [],
}

kernel_m52 = {
    'name' : 'matern52',
    'functions' : L_cov_m52_xi,
    'data' : GP_DATA_GENERIC,
    'transformed_data' : None,
    'parameters' : GP_PARAMETERS_GENERIC,
    'kernel_signature' : 'L_cov_m52_xi(__x__, amp, ls, xi, jitter, n, d)',
    'model' : GP_MODEL_GENERIC,
    'args' : GP_ARGS_GENERIC,
    'to_sample' : ['amp', 'ls'],
    'data_priors' : GP_DATA_PRIORS_GENERIC,
}

kernel_m32 = {
    'name' : 'matern32',
    'functions' : L_cov_m32_xi,
    'data' : GP_DATA_GENERIC,
    'transformed_data' : None,
    'parameters' : GP_PARAMETERS_GENERIC,
    'kernel_signature' : 'L_cov_m32_xi(__x__, amp, ls, xi, jitter, n, d)',
    'model' : GP_MODEL_GENERIC,
    'args' : GP_ARGS_GENERIC,
    'to_sample' : ['amp', 'ls'],
    'data_priors' : GP_DATA_PRIORS_GENERIC,
}

kernel_rq = {
    'name' : 'rq',
    'functions' : L_cov_rq_xi,
    'data' : GP_DATA_GENERIC + [
        'real kappa_loc;',
        'real<lower=0> kappa_scale;'
    ],
    'transformed_data' : None,
    'parameters' : GP_PARAMETERS_GENERIC + [
        'real<lower=0> kappa;'
    ],
    'kernel_signature' : 'L_cov_rq_xi(__x__, amp, kappa, ls, xi, jitter, n)',
    'model' : GP_MODEL_GENERIC + [
        'kappa ~ normal(kappa_loc, kappa_scale);'
    ],
    'args' : GP_ARGS_GENERIC + [
        ('kappa', 1)
    ],
    'to_sample' : ['amp', 'ls', 'kappa'],
    'data_priors' : GP_DATA_PRIORS_GENERIC + [
        'kappa_loc',
        'kappa_scale',
    ],
}

def _apply_warping(_kernel):
    kernel = copy.copy(_kernel)
    kernel['name'] = kernel['name'] + '_warped'
    kernel['data'] = kernel['data'] + [
        'vector[d] alpha_warp_mu;',
        'vector[d] alpha_warp_sigma;',
        'vector[d] beta_warp_mu;',
        'vector[d] beta_warp_sigma;',
    ]
    kernel['parameters'] = kernel['parameters'] + [
        'vector<lower=0>[d] alpha_warp;',
        'vector<lower=0>[d] beta_warp;'
    ]
    kernel['transformed_parameters'] = input_warping
    kernel['kernel_signature'] = _warp_sig(kernel['kernel_signature'])
    kernel['args'] = kernel['args'] + [
        ('alpha_warp', 'd'),
        ('beta_warp', 'd')
    ]
    kernel['to_sample'] = kernel['to_sample'] + [
        'alpha_warp',
        'beta_warp',
    ]
    kernel['data_priors'] = kernel['data_priors'] + [
        'alpha_warp_mu',
        'alpha_warp_sigma',
        'beta_warp_mu',
        'beta_warp_sigma',
    ]
    kernel['model'] = kernel['model'] + [
        'alpha_warp ~ lognormal(alpha_warp_mu, alpha_warp_sigma);',
        'beta_warp ~ lognormal(beta_warp_mu, beta_warp_sigma);',
    ]
    return kernel

def _apply(_kernel):
    kernel = copy.copy(_kernel)
    kernel['kernel_signature'] = _default_sig(kernel['kernel_signature'])
    return kernel

def _warp_sig(sig):
    return sig.replace('__x__', 'x_warped')

def _default_sig(sig):
    return sig.replace('__x__', 'x')

AVAILABLE = {
    'expq' : kernel_expq,
    'expq_flat' : kernel_expq_flat,
    'matern52' : kernel_m52,
    'matern32' : kernel_m32,
    'rq' : kernel_rq,
}
