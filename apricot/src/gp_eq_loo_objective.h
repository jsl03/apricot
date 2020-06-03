// This file is licensed under Version 3.0 of the GNU General Public
// License. See LICENSE for a text of the license.
// -----------------------------------------------------------------------------
#ifndef __GP_EQ_LOO_OBJECTIVE_H_
#define __GP_EQ_LOO_OBJECTIVE_H_

#include <math.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <eigen3/unsupported/Eigen/SpecialFunctions>

#include "kernels.h"
#include "misc.h"

class CVEqKernel {
 private:
  const Eigen::MatrixXd x;
  const Eigen::VectorXd y;
  const double jitter;
  const int n;
  const int d;

 public:
  CVEqKernel(const Eigen::MatrixXd &x, const Eigen::VectorXd &y,
             const double jitter);

  double objective(Eigen::Ref<const Eigen::VectorXd> theta);
};

CVEqKernel::CVEqKernel(const Eigen::MatrixXd &x, const Eigen::VectorXd &y,
                       const double jitter)
    : x(x), y(y), jitter(jitter), n(x.rows()), d(x.cols()) {}

double CVEqKernel::objective(Eigen::Ref<const Eigen::VectorXd> theta) {
  double amp_sq = pow(theta(0), 2);
  double sigma_sq = pow(theta(1), 2);
  Eigen::VectorXd ls_sq = theta.tail(theta.size() - 2).array().square();

  Eigen::LLT<Eigen::MatrixXd> k_chol =
      chol_(covEq_(x, amp_sq, ls_sq, sigma_sq, jitter));

  Eigen::VectorXd k_inv_diag =
      k_chol.solve(Eigen::MatrixXd::Identity(n, n)).diagonal();

  Eigen::VectorXd cv =
      -0.5 * log(2 * M_PI) + (0.5 * (k_inv_diag.array().log())) -
      0.5 * (k_chol.solve(y).array().square() / k_inv_diag.array());

  return -cv.mean();
}

#endif  //__GP_EQ_LOO_OBJECTIVE_H_
