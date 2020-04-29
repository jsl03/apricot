// This file is licensed under Version 3.0 of the GNU General Public
// License. See LICENSE for a text of the license.
// -----------------------------------------------------------------------------
#ifndef __GP_EQ_MLE_OBJECTIVE_H_
#define __GP_EQ_MLE_OBJECTIVE_H_

#include <math.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <eigen3/unsupported/Eigen/SpecialFunctions>
#include <vector>  // necessary to satisfy the linter for some reason

#include "kernels.h"
#include "misc.h"

class NLMLEqKernel {
 private:
  Eigen::MatrixXd x;
  Eigen::VectorXd y;
  double jitter;
  int n;
  int d;
  double term0;

 public:
  NLMLEqKernel(const Eigen::MatrixXd &x, const Eigen::VectorXd &y,
                     const double jitter);

  double objective(Eigen::Ref<const Eigen::VectorXd> theta);
  std::tuple<double, Eigen::VectorXd> objective_jac(
      Eigen::Ref<const Eigen::VectorXd> theta);
};

NLMLEqKernel::NLMLEqKernel(const Eigen::MatrixXd &x,
                                       const Eigen::VectorXd &y,
                                       const double jitter)
    : x(x), y(y), jitter(jitter), n(x.rows()), d(x.cols()) {
  term0 = (n / 2.0) * log(2.0 * M_PI);
}

double NLMLEqKernel::objective(Eigen::Ref<const Eigen::VectorXd> theta) {
  double amp_sq = pow(theta(0), 2);
  double sigma_sq = pow(theta(1), 2);
  Eigen::VectorXd ls_sq = theta.tail(theta.size() - 2).array().square();

  Eigen::LLT<Eigen::MatrixXd> k_chol =
      chol_(covEq_(x, amp_sq, ls_sq, sigma_sq, jitter));

  double loglik = term0;
  loglik += 0.5 * (y.transpose() * k_chol.solve(y))(0);
  loglik += k_chol.matrixLLT().diagonal().array().log().sum();
  return loglik / n;
}

std::tuple<double, Eigen::VectorXd> NLMLEqKernel::objective_jac(
    Eigen::Ref<const Eigen::VectorXd> theta) {
  double amp = theta(0);
  double amp_sq = pow(amp, 2);
  double sigma = theta(1);
  double sigma_sq = pow(sigma, 2);
  Eigen::VectorXd ls = theta.tail(theta.size() - 2);
  Eigen::VectorXd ls_sq = theta.array().square();
  //
  // log marginal likelihood -------------------------------------------------
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(x.rows(), x.rows());
  Eigen::MatrixXd tau = scaledDistanceSelf(x, ls_sq);

  // k(x, x | theta)
  Eigen::MatrixXd k = ((amp_sq * ((-0.5 * tau).array().exp())).array() +
                       sigmaToMat(n, sigma_sq, jitter).array())
                          .matrix()
                          .selfadjointView<Eigen::Lower>();

  // Lower cholesky factor of k
  Eigen::LLT<Eigen::MatrixXd> k_chol = chol_(k);

  // k_inv * y
  Eigen::MatrixXd alpha = k_chol.solve(y);

  // compute the negative log marginal likelihood
  double loglik = term0;
  loglik += 0.5 * (y.transpose() * alpha)(0);
  loglik += k_chol.matrixLLT().diagonal().array().log().sum();
  loglik /= x.rows();

  // gradients  --------------------------------------------------------------
  Eigen::VectorXd jac(theta.size());
  Eigen::MatrixXd w = (alpha * alpha.transpose()) - k_chol.solve(identity);

  // derivative w.r.t amp
  Eigen::MatrixXd dkda =
      ((2.0 * sqrt(amp_sq) * ((-0.5 * tau).array().exp())).array())
          .matrix()
          .selfadjointView<Eigen::Lower>();
  jac(0) = loglikDeriv(w, dkda);

  // derivatives w.r.t. sigma
  Eigen::MatrixXd dkds = Eigen::MatrixXd::Identity(n, n).array() * 2 * sigma;
  jac(1) = loglikDeriv(w, dkds);

  // derivatives w.r.t. ls
  Eigen::MatrixXd dkdl(x.rows(), x.rows());
  for (int idx = 0; idx < d; idx++) {
    dkdl = k.array() * subtractSelfOuter(x.col(idx)).array().square() /
           pow(ls(idx), 3);
    jac(2 + idx) = loglikDeriv(w, dkdl);
  }

  // scale derivatives by 1/n
  jac = jac.array() / x.rows();
  return std::make_tuple(loglik, jac);
}

#endif  //__GP_EQ_MLE_OBJECTIVE_H_
