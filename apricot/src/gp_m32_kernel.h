// This file is licensed under Version 3.0 of the GNU General Public
// License. See LICENSE for a text of the license.
// -----------------------------------------------------------------------------
#ifndef __GP_M32_KERNEL_H_
#define __GP_M32_KERNEL_H_

#include <math.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <eigen3/unsupported/Eigen/SpecialFunctions>
#include <vector>  // necessary to satisfy the linter for some reason

#include "kernels.h"
#include "misc.h"

class GpM32Kernel {
 private:
  Eigen::MatrixXd x;
  Eigen::VectorXd y;
  Eigen::VectorXd amp_sqs;
  Eigen::MatrixXd ls_sqs;
  Eigen::VectorXd sigma_sqs;
  double jitter;
  int n;
  int d;
  int s;
  std::vector<Eigen::LLT<Eigen::MatrixXd>> lxxs;
  std::vector<Eigen::VectorXd> alphas;

 public:
  GpM32Kernel(const Eigen::MatrixXd &xdata, const Eigen::VectorXd &ydata,
              const Eigen::VectorXd &amp, const Eigen::MatrixXd &ls,
              const Eigen::VectorXd &sigma, const double jitter);

  const Eigen::MatrixXd view_x();

  const Eigen::VectorXd view_y();

  std::tuple<const Eigen::VectorXd, const Eigen::MatrixXd,
             const Eigen::VectorXd>
  view_parameters();

  std::vector<Eigen::MatrixXd> view_lxx();

  std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> posterior(
      Eigen::Ref<const Eigen::MatrixXd> xstar);

  std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> posterior_chol(
      Eigen::Ref<const Eigen::MatrixXd> xstar);

  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> marginals(
      Eigen::Ref<const Eigen::MatrixXd> xstar);

  Eigen::VectorXd E(Eigen::Ref<const Eigen::MatrixXd> xstar);

  std::tuple<double, Eigen::VectorXd> E_jac(
      Eigen::Ref<const Eigen::RowVectorXd> xstar);

  Eigen::VectorXd px(Eigen::Ref<const Eigen::MatrixXd> xstar);

  std::tuple<Eigen::VectorXd, Eigen::VectorXd> px_jac(
      Eigen::Ref<const Eigen::RowVectorXd> xstar);

  Eigen::VectorXd ei(Eigen::Ref<const Eigen::MatrixXd> xstar);

  std::tuple<double, Eigen::VectorXd> ei_jac(
      Eigen::Ref<const Eigen::RowVectorXd> xstar);

  Eigen::VectorXd ucb(Eigen::Ref<const Eigen::MatrixXd> xstar, double beta);

  std::tuple<double, Eigen::VectorXd> ucb_jac(
      Eigen::Ref<const Eigen::RowVectorXd> xstar, double beta);

  Eigen::MatrixXd loo_cv();

  double entropy(Eigen::Ref<const Eigen::MatrixXd> xstar);
};

// M32 kernel GP class-constructor
//
// Inputs:
//     xdata : (n,d) array
//         n points in d dimensions
//     ydata : (n) array
//         n scalar responses
//     amp : (s) array
//         a total of s marginal standard deviation hyperparameter samples
//     ls : (s, d) array
//         a total of s length d vectors representing dimensionwise
//         (anisotropic) lengthscale samples
//     sigma : (s) array
//         a total of s Gaussian noise standard deviation hyperparameter
//         samples
//     jitter : double
//         stability jitter
//
// Outputs:
//     GpM32Kernel
//         GP with Matern32 kernel
//
// Notes:
//     Class constructor wants *non-squared* hyperparameters; they are squared
//     on initialising the constructor.
//
GpM32Kernel::GpM32Kernel(const Eigen::MatrixXd &xdata,
                         const Eigen::VectorXd &ydata,
                         const Eigen::VectorXd &amp, const Eigen::MatrixXd &ls,
                         const Eigen::VectorXd &sigma, const double jitter)
    : x(xdata),
      y(ydata),
      amp_sqs(amp.array().square()),
      ls_sqs(ls.array().square()),
      sigma_sqs(sigma.array().square()),
      jitter(jitter),
      n(x.rows()),
      d(x.cols()),
      s(amp.size()) {
  lxxs.reserve(s);
  alphas.reserve(s);
  for (int i = 0; i < s; i++) {
    lxxs.push_back(
        chol_(covM32_(x, amp_sqs(i), ls_sqs.row(i), sigma_sqs(i), jitter)));
    alphas.push_back(lxxs[i].solve(y));
  }
}

// view the data x
//
// Outputs:
//     x : (n, d)
//         n points in d dimensions
//
const Eigen::MatrixXd GpM32Kernel::view_x() { return x; }

// view the data y
//
// Outputs:
//     y : (n) array
//         n scalar responses
//
const Eigen::VectorXd GpM32Kernel::view_y() { return y; }

// view the hyperparameters
//
// Outputs:
//    amp_sq : (s) array
//        a total of s marginal standard deviation hyperparameter samples
//    ls_sq : (s, d) array
//        a total of s length d vectors representing dimensionwise
//        (anisotropic) lengthscale samples
//    sigma_sq : (s) array
//        a total of s Gaussian noise standard deviation hyperparameter
//        samples
//
std::tuple<const Eigen::VectorXd, const Eigen::MatrixXd, const Eigen::VectorXd>
GpM32Kernel::view_parameters() {
  return std::make_tuple(amp_sqs, ls_sqs, sigma_sqs);
}

// view the lower Cholesky factors of k(x, x | amp, ls, sigma)
//
// Outputs:
//     lxxs : list of (n,n) array
//         Length s list of the lower Cholesky factors of the sample covariance
//         matrices computed using the kth set of hyperparameters, that is:
//         lxxs(k) = chol(k(x, x | amp_k, ls_k, sigma_k))
//
std::vector<Eigen::MatrixXd> GpM32Kernel::view_lxx() {
  std::vector<Eigen::MatrixXd> lxx_views;
  lxx_views.reserve(s);
  for (int i = 0; i < s; i++) {
    lxx_views.push_back(lxxs[i].matrixL());
  }
  return lxx_views;
}

// posterior joint distribution at xstar
//
// Inputs:
//     xstar : (m, d) array
//         m points in d dimensions, at which to compute the posterior
//         distribution of the GP
//
// Outputs:
//     means : (m, s) array
//         A total of s length m vectors describing the posterior mean of the GP
//         at the points in xstar for each of the s hyperparameter samples.
//     cov_mats : Length s list of (m, m) arrays
//         A total of s (m, m) arrays describing the posterior covariance
//         structure of the GP at the points in xstar for each of the s
//         hyperparameter samples.
//
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>>
GpM32Kernel::posterior(Eigen::Ref<const Eigen::MatrixXd> xstar) {
  int m = xstar.rows();
  Eigen::MatrixXd means(n, s);
  std::vector<Eigen::MatrixXd> covmats;
  covmats.reserve(s);
  Eigen::MatrixXd kxxstar(n, m);
  for (int i = 0; i < s; i++) {
    kxxstar = crossCovM32_(x, xstar, amp_sqs(i), ls_sqs.row(i));
    means.col(i) = kxxstar.transpose() * alphas[i];
    covmats.push_back(conditionalCov(
        covM32_(xstar, amp_sqs(i), ls_sqs.row(i), sigma_sqs(i), jitter),
        lxxs[i], kxxstar));
  }
  return std::make_tuple(means, covmats);
}

// posterior distribution at xstar (Cholesky parametrisation)
//
// Inputs:
//     xstar : (m, d) array
//         m points in d dimensions, at which to compute the posterior
//         distribution of the GP
//
// Outputs:
//     means : (m, s) array
//         A total of s length m vectors describing the posterior mean of the GP
//         at the points in xstar for each of the s hyperparameter samples.
//     cov_mat_chols : Length s list of (m, m) arrays
//         A total of s (m, m) arrays containing the lower Cholesky factors of
//         the posterior covariance structure of the GP at the points in xstar
//         for each of the s hyperparameter samples.
//
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>>
GpM32Kernel::posterior_chol(Eigen::Ref<const Eigen::MatrixXd> xstar) {
  int m = xstar.rows();
  Eigen::MatrixXd means(n, s);
  std::vector<Eigen::MatrixXd> covmats;
  covmats.reserve(s);
  Eigen::MatrixXd kxxstar(n, m);
  for (int i = 0; i < s; i++) {
    kxxstar = crossCovM32_(x, xstar, amp_sqs(i), ls_sqs.row(i));
    means.col(i) = kxxstar.transpose() * alphas[i];
    covmats.push_back(
        chol_(conditionalCov(covM32_(xstar, amp_sqs(i), ls_sqs.row(i),
                                     sigma_sqs(i), jitter),
                             lxxs[i], kxxstar))
            .matrixL());
  }
  return std::make_tuple(means, covmats);
}

// posterior marginals at xstar
//
// Inputs:
//     xstar : (m, d) array
//         m points in d dimensions, at which to compute the posterior marginals
//         of the GP
//
// Outputs:
//     means : (m, s) array
//         A total of t length m vectors describing the posterior mean of the GP
//         at the points in xstar for each of the s hyperparameter samples.
//     vars : (m, s)
//         A total of s length m vectors describing the posterior marginal
//         variance of the GP at the points in xstar for each of the s
//         hyperparameter samples.
//
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> GpM32Kernel::marginals(
    Eigen::Ref<const Eigen::MatrixXd> xstar) {
  int m = xstar.rows();
  Eigen::MatrixXd means(m, s);
  Eigen::MatrixXd vars(m, s);
  Eigen::MatrixXd kxxstar(n, m);
  for (int i = 0; i < s; i++) {
    kxxstar = crossCovM32_(x, xstar, amp_sqs(i), ls_sqs.row(i));
    means.col(i) = kxxstar.transpose() * alphas[i];
    vars.col(i) =
        conditionalVar(amp_sqs(i) + sigma_sqs(i), lxxs[i], kxxstar, jitter);
  }
  return std::make_tuple(means, vars);
}

// posterior expectation
//
// Inputs:
//     xstar : (m, d) array
//         m points in d dimensions, at which to compute the posterior
//         expectation.
//
// Outputs:
//     expectation : (m) array
//         the posterior expectation at xstar, integrated over the
//         s hyperparameter samples.
//
Eigen::VectorXd GpM32Kernel::E(Eigen::Ref<const Eigen::MatrixXd> xstar) {
  int m = xstar.rows();
  Eigen::VectorXd expectation = Eigen::VectorXd::Zero(m);
  Eigen::MatrixXd kxxstar(n, m);
  for (int i = 0; i < s; i++) {
    kxxstar = crossCovM32_(x, xstar, amp_sqs(i), ls_sqs.row(i));
    expectation.noalias() += (kxxstar.transpose() * alphas[i]);
  }
  return expectation / s;
}

// posterior expectation and jacobian
//
// Inputs:
//     xstar_j : (d) array
//         a single d-dimensional point at which the posterior expectation
//         and its partial derivatives with respect to xstar_j should be
//         computed
//
// Outputs:
//     expectation : double
//         the posterior expectation at xstar_j, integrated over the s
//         hyperparameter samples.
//     expectation_jac : (d) array
//         the partial derivatives of the posterior expectation at xstar_j
//         with respect to xstar_j
//
std::tuple<double, Eigen::VectorXd> GpM32Kernel::E_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar_j) {
  Eigen::RowVectorXd kxstarx(n);
  double expectation = 0;
  Eigen::VectorXd expectation_jac = Eigen::VectorXd::Zero(d);
  for (int i = 0; i < s; i++) {
    kxstarx = crossCovM321d_(xstar_j, x, amp_sqs(i), ls_sqs.row(i));
    expectation += kxstarx.dot(alphas[i]);
    for (int idx = 0; idx < d; idx++) {
      // dE/dx_idx = dK/dx * alpha
      expectation_jac(idx) -=
          crossCovGradM32_(xstar_j, x, idx, amp_sqs(i), ls_sqs.row(i))
              .dot(alphas[i]);
    }
  }
  return std::make_tuple(expectation / s, expectation_jac / s);
}

// pure exploration acquisition function
//
// Inputs:
//     xstar : (m, d) array
//         m points in d dimensions, at which to compute the pure exploration
//         acquisition function
//
// Outputs:
//     px : (m) array
//         the pure exploration at xstar, integrated over the s hyperparameter
//         samples.
//
// Notes:
//     px is negative and equivalent to the (negative) expectation of the
//     marginal variance
//
Eigen::VectorXd GpM32Kernel::px(Eigen::Ref<const Eigen::MatrixXd> xstar) {
  int m = xstar.rows();
  Eigen::VectorXd px = Eigen::VectorXd::Zero(m);
  for (int i = 0; i < s; i++) {
    px.noalias() -= conditionalVar(
        amp_sqs(i) + sigma_sqs(i), lxxs[i],
        crossCovM32_(x, xstar, amp_sqs(i), ls_sqs.row(i)), jitter);
  }
  return px / s;
}

// pure exploration and jacobian
//
// Inputs:
//     xstar_j : (d) array
//         a single d-dimensional point at which the pure exploration
//         acquisition function and its partial derivatives with respect
//         to xstar_j should be computed
//
// Outputs:
//     px : double
//         the pure exploration at xstar_j, integrated over the s
//         hyperparameter samples.
//     px_jac : (d) array
//         the partial derivatives of the pure exploration at xstar_j
//         with respect to xstar_j
//
// Notes:
//     px is negative and equivalent to the (negative) expectation of the
//     marginal variance
//
std::tuple<Eigen::VectorXd, Eigen::VectorXd> GpM32Kernel::px_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar) {
  Eigen::RowVectorXd kxstarx(x.rows());
  Eigen::RowVectorXd px = Eigen::VectorXd::Zero(1);
  Eigen::MatrixXd px_jac = Eigen::VectorXd::Zero(xstar.size());
  for (int i = 0; i < s; i++) {
    kxstarx = crossCovM321d_(xstar, x, amp_sqs(i), ls_sqs.row(i));
    px.noalias() -=
        conditionalVar1d(amp_sqs(i) + sigma_sqs(i), lxxs[i], kxstarx, jitter);
    for (int idx = 0; idx < d; idx++) {
      px_jac(idx) +=
          kxstarx * lxxs[i].solve(-crossCovGradM32_(xstar, x, idx, amp_sqs(i),
                                                    ls_sqs.row(i))
                                       .transpose());
    }
  }
  return std::make_tuple(px / s, px_jac * 2 / s);
}

// expected improvement (EI) acquisition function
//
// Inputs:
//     xstar : (m, d) array
//         m points in d dimensions, at which to compute the expected
//         improvement
//
// Outputs:
//     ei : (m) array
//         the expected improvement at xstar, integrated over the s
//         hyperparameter samples.
//
// Notes:
//     This implementation of EI is negative and seeks the minimum, that is,
//     this function measures the expected reduction over the incumbent minimum.
//
Eigen::VectorXd GpM32Kernel::ei(Eigen::Ref<const Eigen::MatrixXd> xstar) {
  Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
  Eigen::VectorXd ei = Eigen::VectorXd::Zero(xstar.rows());
  double ystar = y.minCoeff();
  for (int i = 0; i < s; i++) {
    kxxstar = crossCovM32_(x, xstar, amp_sqs(i), ls_sqs.row(i));
    ei.noalias() -= expectedImprovementInner_(
        kxxstar.transpose() * alphas[i],
        conditionalVar(amp_sqs(i) + sigma_sqs(i), lxxs[i], kxxstar, jitter)
            .array()
            .sqrt(),
        ystar);
  }
  return ei / s;
}

// expected improvement (EI) and jacobian
//
// Inputs:
//     xstar_j : (d) array
//         a single d-dimensional point at which the expected improvement
//         acquisition function and its partial derivatives with respect
//         to xstar_j should be computed
//
// Outputs:
//     expected_improvement : double
//         the expected improvement at xstar_j, integrated over the s
//         hyperparameter samples.
//     expected_improvement_jac : (d) array
//         the partial derivatives of the expected improvement at xstar_j
//         with respect to xstar_j
//
// Notes:
//     This implementation of EI is negative and seeks the minimum, that is,
//     this function measures the expected reduction over the incumbent minimum.
//
std::tuple<double, Eigen::VectorXd> GpM32Kernel::ei_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar) {
  double ei = 0;
  double mean;
  double mean_grad;
  double var;
  double sd;
  double sd_grad;
  double gamma;
  double gamma_grad;
  double ystar = y.minCoeff();
  Eigen::RowVectorXd kxstarx(n);
  Eigen::RowVectorXd dkxstarx_dxstar(n);
  Eigen::VectorXd ei_jac(d);
  for (int i = 0; i < s; i++) {
    kxstarx = crossCovM321d_(xstar, x, amp_sqs(i), ls_sqs.row(i));
    mean = kxstarx.dot(alphas[i]);
    var = (amp_sqs(i) + sigma_sqs(i)) -
          kxstarx.dot(lxxs[i].solve(kxstarx.transpose()));
    sd = sqrt(var);
    gamma = (ystar - mean) / sd;
    ei -= (ystar - mean) * ncdfd(gamma) + sd * npdfd(gamma);
    for (int idx = 0; idx < d; idx++) {
      dkxstarx_dxstar =
          crossCovGradM32_(xstar, x, idx, amp_sqs(i), ls_sqs.row(i));
      sd_grad =
          -(kxstarx.dot(lxxs[i].solve(-dkxstarx_dxstar.transpose()))) / sd;
      mean_grad = -dkxstarx_dxstar.dot(alphas[i]);
      gamma_grad = (((mean - ystar) * sd_grad) - (sd * mean_grad)) / var;
      ei_jac(idx) -= (ystar - mean) * gamma_grad * npdfd(gamma) -
                     ncdfd(gamma) * mean_grad + npdfd(gamma) * sd_grad +
                     sd * gamma_grad * npdfdg(gamma);
    }
  }
  return std::make_tuple(ei / s, ei_jac / s);
}

// upper confidence bound (UCB) acquisition function
//
// Inputs:
//     xstar : (m, d) array
//         m points in d dimensions, at which to compute the expected
//         improvement
//     beta : double
//         Parameter controlling the trade between exploration and exploitation
//
// Outputs:
//     ucb : (m) array
//         the UCB at xstar, integrated over the s hyperparameter samples.
//
// Notes:
//     This implementation of UCB is negative and seeks the minimum: strictly
//     speaking this function measures a (negative) lower confidence bound.
//
Eigen::VectorXd GpM32Kernel::ucb(Eigen::Ref<const Eigen::MatrixXd> xstar,
                                 double beta) {
  int m = xstar.rows();
  double sr_beta = sqrt(beta);
  Eigen::MatrixXd kxxstar(n, m);
  Eigen::VectorXd ucb = Eigen::VectorXd::Zero(m);
  for (int i = 0; i < s; i++) {
    kxxstar = crossCovM32_(x, xstar, amp_sqs(i), ls_sqs.row(i));
    ucb.noalias() += upperConfidenceBoundInner_(
        kxxstar.transpose() * alphas[i],
        conditionalVar(amp_sqs(i) + sigma_sqs(i), lxxs[i], kxxstar, jitter)
            .array()
            .sqrt(),
        sr_beta);
  }
  return ucb / s;
}

// expected improvement (EI) and jacobian
//
// Inputs:
//     xstar_j : (d) array
//         a single d-dimensional point at which the expected improvement
//         acquisition function and its partial derivatives with respect
//         to xstar_j should be computed
//     beta : double
//         Parameter controlling the trade between exploration and exploitation
//
// Outputs:
//     expected_improvement : double
//         the expected improvement at xstar_j, integrated over the s
//         hyperparameter samples.
//     expected_improvement_jac : (d) array
//         the partial derivatives of the expected improvement at xstar_j
//         with respect to xstar_j
//
// Notes:
//     This implementation of UCB is negative and seeks the minimum: strictly
//     speaking this function measures a (negative) lower confidence bound.
//
std::tuple<double, Eigen::VectorXd> GpM32Kernel::ucb_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar, double beta) {
  double ucb = 0;
  double sd;
  double sr_beta = sqrt(beta);
  Eigen::RowVectorXd kxstarx(n);
  Eigen::RowVectorXd dkxstarx_dxstar(n);
  Eigen::VectorXd ucb_jac(d);
  for (int i = 0; i < s; i++) {
    kxstarx = crossCovM321d_(xstar, x, amp_sqs(i), ls_sqs.row(i));
    sd = sqrt((amp_sqs(i) + sigma_sqs(i)) -
              kxstarx.dot(lxxs[i].solve(kxstarx.transpose())));
    ucb += kxstarx.dot(alphas[i]) - (sr_beta * sd);
    for (int idx = 0; idx < d; idx++) {
      dkxstarx_dxstar =
          crossCovGradM32_(xstar, x, idx, amp_sqs(i), ls_sqs.row(i));
      ucb_jac(idx) +=
          -dkxstarx_dxstar.dot(alphas[i]) -
          (sr_beta *
           -(kxstarx.dot(lxxs[i].solve(-dkxstarx_dxstar.transpose()))) / sd);
    }
  }
  return std::make_tuple(ucb / s, ucb_jac / s);
}

// leave-one-out predictive sample data density
//
// Outputs:
//     loo_cv : (n, t) array
//     loo-cv scores for each of the n training samples
//     for each of the s hyperparameter samples
//
// Notes:
//     Uses the method of Sundararajan and Keerthi:
//     Predictive approaches for choosing hyperparameters in Gaussian Processes.
//     Advances in Neural Information Processing Systems 631-637
//     2000
//
Eigen::MatrixXd GpM32Kernel::loo_cv() {
  Eigen::MatrixXd cv(s, n);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd c_diag;
  for (int i = 0; i < s; i++) {
    c_diag = lxxs[i].solve(I).diagonal();
    cv.row(i) = -0.5 * log(2 * M_PI) + (0.5 * (c_diag.array().log())) -
                0.5 * (alphas[i].array().square() / c_diag.array());
  }
  return cv;
}

// differential entropy
//
// Inputs:
//     xstar : (m, d) array
//         m points in d dimensions, at which to compute the differential
//         entropy
//
// Outputs:
//     h : The differential entropy of the GP posterior calculated at the
//         points containing in xstar, integrated over the s hyperparameter
//         samples
//
double GpM32Kernel::entropy(Eigen::Ref<const Eigen::MatrixXd> xstar) {
  int m = xstar.rows();
  const double e = std::exp(1.0);
  double h = 0;
  double tmp = n / 2.0 * log(e * M_PI * 2.0);
  Eigen::MatrixXd kxxstar(n, m);
  Eigen::MatrixXd c(m, m);
  for (int i = 0; i < s; i++) {
    kxxstar = crossCovM32_(x, xstar, amp_sqs(i), ls_sqs.row(i));
    // computes
    //
    // h = tmp + 0.5 * 2 * log(det(Sigma))
    // = tmp + 0.5 * 2 * log(det(kzz - kzx * kxx_inv * kxz))
    // = tmp + sum(log(diag(chol(kzz - kzx * kxx_inv * kxz))))
    //
    // in a way that Eigen can (hopefully) optimise
    h += tmp + (covM32_(xstar, amp_sqs(i), ls_sqs.row(i), sigma_sqs(i), jitter)
                    .array() -
                ((kxxstar.transpose() * lxxs[i].solve(kxxstar))).array())
                   .matrix()
                   .llt()
                   .matrixLLT()
                   .diagonal()
                   .array()
                   .log()
                   .sum();
  }
  return h / s;
}
#endif  // __GP_M32_INTERNAL_H_
