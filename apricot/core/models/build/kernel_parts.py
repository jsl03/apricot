# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
import copy
from typing import Dict, Any, Union
from apricot.core.models.build.components import StanModelKernel
from apricot.core.models.build.code_snippets import L_COV_EQ_SIGMA
from apricot.core.models.build.code_snippets import L_COV_M52_SIGMA
from apricot.core.models.build.code_snippets import L_COV_M32_SIGMA
from apricot.core.models.build.code_snippets import L_COV_RQ_SIGMA
from apricot.core.models.build.code_snippets import INPUT_WARPING


def find_kernel(name: str, warping: Union[bool, str]) -> Dict[str, Any]:
    """ Find named kernel and apply supplied warping option. """
    kernel_options = AVAILABLE[name]
    if warping:
        return apply_warping(kernel_options)
    return apply(kernel_options)


def make_kernel(
        kernel_type: str,
        warping: Union[bool, str]
) -> StanModelKernel:
    """ Find requested kernel and return Stan model part. """
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


KERNEL_EQ = {
    'name': 'eq',
    'functions': L_COV_EQ_SIGMA,
    'data': GP_DATA_GENERIC,
    'transformed_data': None,
    'parameters': GP_PARAMETERS_GENERIC,
    'kernel_signature': 'L_cov_eq_sigma(__x__, amp, ls, sigma, jitter, n)',
    'model': GP_MODEL_GENERIC,
    'args': GP_ARGS_GENERIC,
    'to_sample': ['amp', 'ls'],
    'data_priors': GP_DATA_PRIORS_GENERIC,
}


KERNEL_EQ_FLAT = {
    'name': 'eq_flat',
    'functions': L_COV_EQ_SIGMA,
    'data': [],
    'transformed_data': None,
    'parameters': GP_PARAMETERS_GENERIC,
    'kernel_signature': 'L_cov_eq_sigma(__x__, amp, ls, sigma, jitter, n)',
    'model': [],
    'args': GP_ARGS_GENERIC,
    'to_sample': ['amp', 'ls'],
    'data_priors': [],
}


KERNEL_M52 = {
    'name': 'm52',
    'functions': L_COV_M52_SIGMA,
    'data': GP_DATA_GENERIC,
    'transformed_data': None,
    'parameters': GP_PARAMETERS_GENERIC,
    'kernel_signature': 'L_cov_m52_sigma(__x__, amp, ls, sigma, jitter, n, d)',
    'model': GP_MODEL_GENERIC,
    'args': GP_ARGS_GENERIC,
    'to_sample': ['amp', 'ls'],
    'data_priors': GP_DATA_PRIORS_GENERIC,
}


KERNEL_M32 = {
    'name': 'm32',
    'functions': L_COV_M32_SIGMA,
    'data': GP_DATA_GENERIC,
    'transformed_data': None,
    'parameters': GP_PARAMETERS_GENERIC,
    'kernel_signature': 'L_cov_m32_sigma(__x__, amp, ls, sigma, jitter, n, d)',
    'model': GP_MODEL_GENERIC,
    'args': GP_ARGS_GENERIC,
    'to_sample': ['amp', 'ls'],
    'data_priors': GP_DATA_PRIORS_GENERIC,
}


KERNEL_RQ = {
    'name': 'rq',
    'functions': L_COV_RQ_SIGMA,
    'data': GP_DATA_GENERIC + [
        'real kappa_loc;',
        'real<lower=0> kappa_scale;'
    ],
    'transformed_data': None,
    'parameters': GP_PARAMETERS_GENERIC + [
        'real<lower=0> kappa;'
    ],
    'kernel_signature': 'L_cov_rq_sigma(__x__, amp, kappa, ls, sigma, jitter, n)',
    'model': GP_MODEL_GENERIC + [
        'kappa ~ normal(kappa_loc, kappa_scale);'
    ],
    'args': GP_ARGS_GENERIC + [
        ('kappa', 1)
    ],
    'to_sample': ['amp', 'ls', 'kappa'],
    'data_priors': GP_DATA_PRIORS_GENERIC + [
        'kappa_loc',
        'kappa_scale',
    ],
}


def apply_warping(_kernel: Dict[str, Any]) -> Dict[str, Any]:
    """ Adds generic code for input warping into the required data fields.

    Summary of Stan model code changes:

    Name:
        Appends "_warped"

    Data:
        Adds data fields:
        * alpha_warp_mu
            Lognormal distribution mean for warping parameter alpha
        * alpha_warp_sigma
            Lognormal distribution s.d. for warping parameter alpha
        * beta_warp_mu
            Lognormal distribution mean for warping parameter beta
        * beta_warp_sigma
            Lognormal distribution s.d. for warping parameter beta

    Parameters:
        Adds parameters:
        * alpha_warp
        * beta_warp

    Transformed Parameters:
        Adds input warping transformation x -> x_warped via Beta CDF
        (see Snoek et. al. 2014)

    Model:
        Adds:
        * alpha_warp ~ lognormal(alpha_warp_mu, alpha_warp_sigma)
        * beta_warp ~ lognormal(beta_warp_mu, beta_warp_sigma)
    """
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
    kernel['transformed_parameters'] = INPUT_WARPING
    kernel['kernel_signature'] = warp_sig(kernel['kernel_signature'])
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


def apply(_kernel: Dict[str, Any]) -> Dict[str, Any]:
    """ Apply the correct function signature to the covariance kernel.

    If the kernel features warping, change the function signature of
    the call to the covariance kernel to use "x_warped".

    Otherwise, change the function signature to use "x".
    """
    kernel = copy.copy(_kernel)
    kernel['kernel_signature'] = _default_sig(kernel['kernel_signature'])
    return kernel


def warp_sig(sig: str) -> str:
    """ Replace __x__ with x_warped in the covariance kernel call."""
    return sig.replace('__x__', 'x_warped')


def _default_sig(sig: str) -> str:
    """ Replace __x__ with x in the covariance kernel call."""
    return sig.replace('__x__', 'x')


AVAILABLE = {
    'eq': KERNEL_EQ,
    'eq_flat': KERNEL_EQ_FLAT,
    'm52': KERNEL_M52,
    'm32': KERNEL_M32,
    'rq': KERNEL_RQ,
}
