"""
pyStan model code snippets as strings.
"""

x_to_matrix = '\n'.join([
    'matrix[n, d + 1] X_matrix ;',
    'for (i in 1:n) {',
    '  X_matrix[i, 1] = 1;',
    '  for (j in 1:d) {',
    '    X_matrix[i, 1+j] = x[i][j];',
    '  }',
    '}',
])

x_dot_beta = 'mu = X_matrix * beta;'

mu_to_zeros = 'mu = rep_vector(0,n);'

L_cov_expq_xi = '\n'.join([
    'matrix L_cov_expq_xi(vector[] x, real amp, vector ls, real xi, real jitter, int n) {',
    '  matrix[n, n] C;',
    '  real tmp;',
    '  real amp_sq = square(amp);',
    '  real xi_sq = square(xi);',
    '  for (i in 1:n) {',
    '    for (j in 1:i) {',
    '      if (i==j){ C[i, i] = amp_sq + xi_sq + jitter;}',
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

L_cov_rq_xi = '\n'.join([
    'matrix L_cov_rq_xi(vector[] x, real amp, real kappa, vector ls, real xi, real jitter, int n) {',
    '  matrix[n, n] C;',
    '  real tmp;',
    '  real amp_sq = square(amp);',
    '  real xi_sq = square(xi);',
    '  for (i in 1:n) {',
    '    for (j in 1:i) {',
    '      if (i==j){ C[i, i] = amp_sq + xi_sq + jitter;}',
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

L_cov_m52_xi = '\n'.join([
    'matrix L_cov_m52_xi(vector[] x, real amp, vector ls, real xi, real jitter, int n, int d) {',
    '  matrix[n, n] C;',
    '  real amp_sq = square(amp);',
    '  real xi_sq = square(xi);',
    '  real sqrt5 = sqrt(5.0);',
    '  real r_sq;',
    '  real r;',
    '  real tmp;',
    '  for (i in 1:n) {',
    '    for (j in 1:i) {',
    '      if (i==j) {C[i,i] = amp_sq + xi_sq + jitter;}',
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

L_cov_m32_xi = '\n'.join([
    'matrix L_cov_m32_xi(vector[] x, real amp, vector ls, real xi, real jitter, int n, int d) {',
    '  matrix[n, n] C;',
    '  real amp_sq = square(amp);',
    '  real xi_sq = square(xi);',
    '  real sqrt3 = sqrt(3.0);',
    '  real r_sq;',
    '  real r;',
    '  real tmp;',
    '  for (i in 1:n) {',
    '    for (j in 1:i) {',
    '      if (i==j) {C[i,i] = amp_sq + xi_sq + jitter;}',
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

input_warping = '\n'.join([
'vector[d] x_warped[n];',
'for (i in 1:n) {',
'  for (j in 1:d) {',
'    x_warped[i][j] = beta_cdf(x[i][j], alpha_warp[j], beta_warp[j]);',
'  }',
'}',
])
