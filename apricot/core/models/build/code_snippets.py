# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------

X_TO_MATRIX = '\n'.join([
    'matrix[n, d + 1] X_matrix ;',
    'for (i in 1:n) {',
    '  X_matrix[i, 1] = 1;',
    '  for (j in 1:d) {',
    '    X_matrix[i, 1+j] = x[i][j];',
    '  }',
    '}',
])

X_DOT_BETA = 'mu = X_matrix * beta;'

MU_TO_ZEROS = 'mu = rep_vector(0,n);'

L_COV_EQ_SIGMA = '\n'.join([
    'matrix L_cov_eq_sigma(vector[] x, real amp, vector ls, real sigma, real jitter, int n) {',
    '  matrix[n, n] C;',
    '  real tmp;',
    '  real amp_sq = square(amp);',
    '  real sigma_sq = square(sigma);',
    '  for (i in 1:n) {',
    '    for (j in 1:i) {',
    '      if (i==j){ C[i, i] = amp_sq + sigma_sq + jitter;}',
    '      if (i!=j){',
    '        tmp = amp_sq * exp( -0.5 * dot_self((x[i] - x[j]) ./ ls));',
    '        C[i, j] = tmp;',
    '        C[j, i] = tmp;',
    '      }',
    '    }',
    '  }',
    '  return cholesky_decompose(C) ;',
    '}',
])

L_COV_RQ_SIGMA = '\n'.join([
    'matrix L_cov_rq_sigma(vector[] x, real amp, real kappa, vector ls, real sigma, real jitter, int n) {',
    '  matrix[n, n] C;',
    '  real tmp;',
    '  real amp_sq = square(amp);',
    '  real sigma_sq = square(sigma);',
    '  for (i in 1:n) {',
    '    for (j in 1:i) {',
    '      if (i==j){ C[i, i] = amp_sq + sigma_sq + jitter;}',
    '      if (i!=j){',
    '        tmp = amp_sq * pow( 1 + 0.5 * dot_self((x[i] - x[j]) ./ ls) / kappa, -kappa);',
    '        C[i, j] = tmp;',
    '        C[j, i] = tmp;',
    '      }',
    '    }',
    '  }',
    '  return cholesky_decompose(C);',
    '}',
])


# TODO: vectorise inner r calculation
L_COV_M52_SIGMA = '\n'.join([
    'matrix L_cov_m52_sigma(vector[] x, real amp, vector ls, real sigma, real jitter, int n, int d) {',
    '  matrix[n, n] C;',
    '  real amp_sq = square(amp);',
    '  real sigma_sq = square(sigma);',
    '  real sqrt5 = sqrt(5.0);',
    '  real r_sq;',
    '  real r;',
    '  real tmp;',
    '  for (i in 1:n) {',
    '    for (j in 1:i) {',
    '      if (i==j) {C[i,i] = amp_sq + sigma_sq + jitter;}',
    '      if (i!=j) {',
    '        r_sq = 0;',
    '        for (dim in 1:d) {',
    '          r_sq += square(x[i][dim] - x[j][dim]) / square(ls[dim]);',
    '        }',
    '        r = sqrt(r_sq);',
    '        tmp = amp_sq*(1+sqrt5*r+(5.0/3.0)*r_sq)*exp(-sqrt5*r);',
    '        C[i, j] = tmp;',
    '        C[j, i] = tmp;',
    '      }',
    '    }',
    '  }',
    '  return cholesky_decompose(C);',
    '}',
])

# TODO: vectorise inner r calculation
L_COV_M32_SIGMA = '\n'.join([
    'matrix L_cov_m32_sigma(vector[] x, real amp, vector ls, real sigma, real jitter, int n, int d) {',
    '  matrix[n, n] C;',
    '  real amp_sq = square(amp);',
    '  real sigma_sq = square(sigma);',
    '  real sqrt3 = sqrt(3.0);',
    '  real r_sq;',
    '  real r;',
    '  real tmp;',
    '  for (i in 1:n) {',
    '    for (j in 1:i) {',
    '      if (i==j) {C[i,i] = amp_sq + sigma_sq + jitter;}',
    '      if (i!=j) {',
    '        r_sq = 0;',
    '        for (dim in 1:d) {',
    '          r_sq += square(x[i][dim] - x[j][dim]) / square(ls[dim]);',
    '        }',
    '        r = sqrt(r_sq);',
    '        tmp = amp_sq*(1+sqrt3*r)*exp(-sqrt3*r);',
    '        C[i, j] = tmp;',
    '        C[j, i] = tmp;',
    '      }',
    '    }',
    '  }',
    '  return cholesky_decompose(C);',
    '}',
])

INPUT_WARPING = '\n'.join([
    'vector[d] x_warped[n];',
    'for (i in 1:n) {',
    '  for (j in 1:d) {',
    '    x_warped[i][j] = beta_cdf(x[i][j], alpha_warp[j], beta_warp[j]);',
    '  }',
    '}',
])
