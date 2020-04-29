// This file is licensed under Version 3.0 of the GNU General Public
// License. See LICENSE for a text of the license.
// -----------------------------------------------------------------------------
#ifndef __KERNELS_H_
#define __KERNELS_H_

#include <math.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <eigen3/unsupported/Eigen/SpecialFunctions>

#include "misc.h"

// Exponentiated Quadratic (EQ) Kernel -----------------------------------------

// EQ kernel sample covariance matrix
//
// Inputs:
//     x : (n, d) array
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise squared lengthscales
//     sigma_sq : double
//         squared Gaussian noise standard deviation (noise variance)
//
// Outputs:
//     kxx : (n, n) array
//         Covariance matrix cov(x, x | amp_sq, ls_sq, sigma_sq)
//
Eigen::MatrixXd covEq_(const Eigen::MatrixXd &x, const double amp_sq,
                       const Eigen::VectorXd &ls_sq, const double sigma_sq,
                       const double jitter) {
  int n = x.rows();
  Eigen::MatrixXd tau = scaledDistanceSelf(x, ls_sq);
  return ((amp_sq * ((-0.5 * tau).array().exp())).array() +
          sigmaToMat(n, sigma_sq, jitter).array())
      .matrix()
      .selfadjointView<Eigen::Lower>();
}

// EQ kernel cross covariance matrix
//
// Inputs:
//     x1 : (n, d) array
//         n points in d dimensions
//     x2 : (m, d)
//         m different points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//
// Outputs:
//     kx1x2 : (n, m) array
//         Cross covariance matrix k(x1, x2 | amp_sq, ls_sq)
//
Eigen::MatrixXd crossCovEq_(const Eigen::MatrixXd &x1,
                            const Eigen::MatrixXd &x2, const double amp_sq,
                            const Eigen::VectorXd &ls_sq) {
  return amp_sq * ((-0.5 * scaledDistance(x1, x2, ls_sq)).array().exp());
}

// EQ kernel cross covariance between a vector and a matrix
//
// Inputs:
//     xstar_j : (d) array
//         a single d dimensional point
//     x : (n, d)
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//
// Outputs:
//     kxstar_jx : (n) array
//         Cross covariance k(xstar_j, x | amp_sq, ls_sq)
//
Eigen::VectorXd crossCovEq1d_(const Eigen::VectorXd &xstar_j,
                              const Eigen::MatrixXd &x, const double amp_sq,
                              const Eigen::VectorXd &ls_sq) {
  return amp_sq * ((-0.5 * scaledDistance1d(xstar_j, x, ls_sq)).array().exp());
}

// EQ kernel 1d cross covariance partial derivatives
//
// Inputs:
//     xstar_j : (d) array
//         a single d dimensional point
//     x : (n, d) array
//         n points in d dimensions
//     idx : int
//         dimension in which to compute the partial derivative
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//     kxstarx : (n) array
//         Output of cov_1d_eq(xstar_j, x, amp_sq, ls_sq)
//
// Outputs:
//     dk : (n) array
//         partial derivatives of k(xstar_j, x | amp_sq, ls_sq)
//         with respect to xstar_j in dimension d
//
Eigen::RowVectorXd crossCovGradEq_(const Eigen::VectorXd &xstar_j,
                                   const Eigen::MatrixXd &x, const int idx,
                                   const Eigen::VectorXd &ls_sq,
                                   const Eigen::VectorXd &kxstarx) {
  return xstarMinusxOverls_(xstar_j(idx), x.col(idx), ls_sq(idx)).array() *
         kxstarx.array();
}

// Matern52 (M52) Kernel -------------------------------------------------------

// Matern52 kernel sample covariance matrix
//
// Inputs:
//     x : (n, d) array
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise squared lengthscales
//     sigma_sq : double
//         squared Gaussian noise standard deviation (noise variance)
//
// Outputs:
//     kxx : (n, n) array
//         Covariance matrix cov(x, x | amp_sq, ls_sq, sigma_sq)
//
Eigen::MatrixXd covM52_(const Eigen::MatrixXd &x, const double amp_sq,
                        const Eigen::VectorXd &ls_sq, const double sigma_sq,
                        const double jitter) {
  int n = x.rows();
  double sr_5 = sqrt(5.0);
  Eigen::MatrixXd tau = scaledDistanceSelf(x, ls_sq);
  Eigen::MatrixXd sr_tau = tau.array().sqrt();
  return (amp_sq * (1.0 + sr_5 * sr_tau.array() + (5.0 / 3.0) * tau.array()) *
              (-sr_5 * sr_tau).array().exp().array() +
          sigmaToMat(n, sigma_sq, jitter).array())
      .matrix()
      .selfadjointView<Eigen::Lower>();
}

// Matern52 kernel cross covariance matrix
//
// Inputs:
//     x1 : (n, d) array
//         n points in d dimensions
//     x2 : (m, d)
//         m different points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//
// Outputs:
//     kx1x2 : (n, m) array
//         Cross covariance matrix k(x1, x2 | amp_sq, ls_sq)
//
Eigen::MatrixXd crossCovM52_(const Eigen::MatrixXd &x1,
                             const Eigen::MatrixXd &x2, const double amp_sq,
                             const Eigen::VectorXd &ls_sq) {
  double sr_5 = sqrt(5.0);
  Eigen::MatrixXd tau = scaledDistance(x1, x2, ls_sq);
  Eigen::MatrixXd sr_tau = tau.array().sqrt();
  return amp_sq * (1.0 + sr_5 * sr_tau.array() + (5.0 / 3.0) * tau.array()) *
         (-sr_5 * sr_tau).array().exp();
}

// M52 kernel cross covariance between a vector and a matrix
//
// Inputs:
//     xstar_j : (d) array
//         a single d dimensional point
//     x : (n, d)
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//
// Outputs:
//     kxstar_jx : (n) array
//         Cross covariance k(xstar_j, x | amp_sq, ls_sq)
//
Eigen::VectorXd crossCovM521d_(const Eigen::VectorXd &xstar_j,
                               const Eigen::MatrixXd &x, const double amp_sq,
                               const Eigen::VectorXd &ls_sq) {
  double sr_5 = sqrt(5.0);
  Eigen::VectorXd tau = scaledDistance1d(xstar_j, x, ls_sq);
  Eigen::VectorXd sr_tau = tau.array().sqrt();
  return amp_sq * (1.0 + sr_5 * sr_tau.array() + (5.0 / 3.0) * tau.array()) *
         (-sr_5 * sr_tau).array().exp();
}

// Matern52 kernel 1d cross covariance partial derivatives
//
// Inputs:
//     xstar_j : (d) array
//         a single d dimensional point
//     x : (n, d) array
//         n points in d dimensions
//     d : int
//         dimension in which to compute the partial derivative
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//     kxstarx : (n) array
//         Output of cov_1d_eq(xstar_j, x, amp_sq, ls_sq)
//
// Outputs:
//     dk : (n) array
//         partial derivatives of k(xstar_j, x | amp_sq, ls_sq)
//         with respect to xstar_j in dimension d
//
Eigen::RowVectorXd crossCovGradM52_(const Eigen::VectorXd &xstar_j,
                                    const Eigen::MatrixXd &x, const int idx,
                                    const double amp_sq,
                                    const Eigen::VectorXd &ls_sq) {
  Eigen::VectorXd tau = scaledDistance1d(xstar_j, x, ls_sq).array().sqrt() * sqrt(5.0);
  return -(xstarMinusxOverls_(xstar_j(idx), x.col(idx), ls_sq(idx))).array() *
         (amp_sq * ((10.0 / 3.0) - 5.0 - (5.0 / 3.0) * tau.array()) *
          (-1 * tau).array().exp());
}

// Matern32 (M32) Kernel -------------------------------------------------------

// Matern32 kernel sample covariance matrix
//
// Inputs:
//     x : (n, d) array
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise squared lengthscales
//     sigma_sq : double
//         squared Gaussian noise standard deviation (noise variance)
//
// Outputs:
//     kxx : (n, n) array
//         Covariance matrix cov(x, x | amp_sq, ls_sq, sigma_sq)
//
Eigen::MatrixXd covM32_(const Eigen::MatrixXd &x, const double amp_sq,
                        const Eigen::VectorXd &ls_sq, const double sigma_sq,
                        const double jitter) {
  int n = x.rows();
  Eigen::MatrixXd tau = scaledDistanceSelf(x, ls_sq);
  tau = tau.array().sqrt() * sqrt(3.0);
  return ((amp_sq * (1.0 + tau.array()) * (-tau).array().exp()).array() +
          sigmaToMat(n, sigma_sq, jitter).array())
      .matrix()
      .selfadjointView<Eigen::Lower>();
}

// Matern32 kernel cross covariance matrix
//
// Inputs:
//     x1 : (n, d) array
//         n points in d dimensions
//     x2 : (m, d)
//         m different points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         diMensionwise (anisotropic) squared lengthscales
//
// Outputs:
//     kx1x2 : (n, m) array
//         Cross covariance matrix k(x1, x2 | amp_sq, ls_sq)
//
Eigen::MatrixXd crossCovM32_(const Eigen::MatrixXd &x1,
                             const Eigen::MatrixXd &x2, const double amp_sq,
                             const Eigen::VectorXd &ls_sq) {
  Eigen::MatrixXd tau = scaledDistance(x1, x2, ls_sq);
  tau = tau.array().sqrt() * sqrt(3);
  return amp_sq * (1.0 + tau.array()) * (-tau).array().exp();
}

// M32 kernel cross covariance between a vector and a matrix
//
// Inputs:
//     xstar_j : (d) array
//         a single d dimensional point
//     x : (n, d)
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//
// Outputs:
//     kxstar_jx : (n) array
//         Cross covariance k(xstar_j, x | amp_sq, ls_sq)
//
Eigen::VectorXd crossCovM321d_(const Eigen::VectorXd &xstar_j,
                               const Eigen::MatrixXd &x, const double amp_sq,
                               const Eigen::VectorXd &ls_sq) {
  Eigen::VectorXd tau = scaledDistance1d(xstar_j, x, ls_sq);
  tau = tau.array().sqrt() * sqrt(3);
  return amp_sq * (1.0 + tau.array()) * (-tau).array().exp();
}

// Matern32 kernel 1d cross covariance partial derivatives
//
// Inputs:
//     xstar_j : (d) array
//         a single d dimensional point
//     x : (n, d) array
//         n points in d dimensions
//     d : int
//         dimension in which to compute the partial derivative
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//     kxstarx : (n) array
//         Output of cov_1d_eq(xstar_j, x, amp_sq, ls_sq)
//
// Outputs:
//     dk : (n) array
//         partial derivatives of k(xstar_j, x | amp_sq, ls_sq)
//         with respect to xstar_j in dimension d
//
Eigen::RowVectorXd crossCovGradM32_(const Eigen::VectorXd &xstar_j,
                                    const Eigen::MatrixXd &x, const int idx,
                                    const double amp_sq,
                                    const Eigen::VectorXd &ls_sq) {
  return -xstarMinusxOverls_(xstar_j(idx), x.col(idx), ls_sq(idx)).array() *
         (amp_sq * -3.0 *
          exp(-(scaledDistance1d(xstar_j, x, ls_sq).array().sqrt() * sqrt(3))));
}

// Rational Quadratic (RQ) Kernel-----------------------------------------------

// RQ kernel sample covariance matrix
//
// Inputs:
//     x : (n, d) array
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise squared lengthscales
//     kappa : double
//         relative scale weighting hyperparameter
//     sigma_sq : double
//         squared Gaussian noise standard deviation (noise variance)
//     jitter : double
//         stability jitter
//
// Outputs:
//     kxx : (n, n) array
//         Covariance matrix cov(x, x | amp_sq, ls_sq, kappa, sigma_sq)
//
Eigen::MatrixXd covRq_(const Eigen::MatrixXd &x, const double amp_sq,
                       const Eigen::VectorXd &ls_sq, const double kappa,
                       const double sigma_sq, const double jitter) {
  int n = x.rows();
  Eigen::MatrixXd tau = scaledDistanceSelf(x, ls_sq);
  return ((amp_sq * pow(1.0 + 0.5 * tau.array() / kappa, -kappa)).array() +
          sigmaToMat(n, sigma_sq, jitter).array())
      .matrix()
      .selfadjointView<Eigen::Lower>();
}

// RQ kernel cross covariance matrix
//
// Inputs:
//     x1 : (n, d) array
//         n points in d dimensions
//     x2 : (m, d)
//         m different points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//     kappa : double
//         relative scale weighting hyperparameter
//
// Outputs:
//     kx1x2 : (n, m) array
//         Cross covariance matrix k(x1, x2 | amp_sq, ls_sq, kappa)
//
Eigen::MatrixXd crossCovRq_(const Eigen::MatrixXd &x1,
                            const Eigen::MatrixXd &x2, const double amp_sq,
                            const Eigen::VectorXd &ls_sq, const double kappa) {
  return amp_sq *
         pow(1.0 + 0.5 * scaledDistance(x1, x2, ls_sq).array() / kappa, -kappa);
}

// RQ kernel cross covariance between a vector and a matrix
//
// Inputs:
//     xstar_j : (d) array
//         a single d dimensional point
//     x : (n, d)
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//
// Outputs:
//     kxstar_jx : (n) array
//         Cross covariance k(xstar_j, x | amp_sq, ls_sq)
//
Eigen::VectorXd crossCovRq1d_(const Eigen::VectorXd &xstar_j,
                              const Eigen::MatrixXd &x, const double amp_sq,
                              const Eigen::VectorXd &ls_sq,
                              const double kappa) {
  return amp_sq *
         pow(1.0 + 0.5 * scaledDistance1d(xstar_j, x, ls_sq).array() / kappa,
             -kappa);
}

// RQ kernel 1d cross covariance partial derivatives
//
// Inputs:
//     xstar_j : (d) array
//         a single d dimensional point
//     x : (n, d) array
//         n points in d dimensions
//     idx : int
//         dimension in which to compute the partial derivative
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//     kappa : double
//         relative scale weighting hyperparameter
//
// Outputs:
//     dk : (n) array
//         partial derivatives of k(xstar_j, x | amp_sq, ls_sq, kappa)
//         with respect to xstar_j in dimension d
//
Eigen::RowVectorXd crossCovGradRq_(const Eigen::VectorXd &xstar_j,
                                   const Eigen::MatrixXd &x, int idx,
                                   double amp_sq, const Eigen::VectorXd &ls_sq,
                                   double kappa) {
  return -xstarMinusxOverls_(xstar_j(idx), x.col(idx), ls_sq(idx)).array() *
         -amp_sq *
         pow(1.0 + 0.5 * scaledDistance1d(xstar_j, x, ls_sq).array() / kappa,
             -kappa - 1.0);
}
#endif  // __KERNELS_H_
