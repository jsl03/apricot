#ifndef __MISC_H_
#define __MISC_H_

#include <math.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>
#include <eigen3/unsupported/Eigen/SpecialFunctions>

// Vectorised Standard Normal PDF
Eigen::VectorXd npdf(const Eigen::VectorXd& x) {
  return (1 / sqrt(2 * M_PI)) * (-(x.array().square() / 2)).exp();
}

// Vectorised Standard Normal CDF
Eigen::VectorXd ncdf(const Eigen::VectorXd& x) {
  return 0.5 * (1 + erf(x.array() / (sqrt(2))));
}

// Standard Normal PDF
double npdfd(double x) { return (1 / sqrt(2 * M_PI)) * exp(-(pow(x, 2) / 2)); }

// Standard Normal CDF
double ncdfd(double x) { return 0.5 * (1 + erf(x / (sqrt(2)))); }

// Derivative of standard Normal PDF with respect to x
double npdfdg(double x) {
  return -(x * exp(-((pow(x, 2)) / 2))) / sqrt(2 * M_PI);
}

/* Conditional covariance matrix */
Eigen::MatrixXd conditionalCov(
    const Eigen::Ref<const Eigen::MatrixXd>& kxstarxstar,
    const Eigen::LLT<Eigen::MatrixXd>& lxx,
    const Eigen::Ref<const Eigen::MatrixXd>& kxxstar) {
  return (kxstarxstar - kxxstar.transpose() * lxx.solve(kxxstar))
      .selfadjointView<Eigen::Lower>();
}

/* Cholesky factor of conditional covariance matrix */
Eigen::MatrixXd conditionalCovChol(
    const Eigen::Ref<const Eigen::MatrixXd>& kxstarxstar,
    const Eigen::LLT<Eigen::MatrixXd>& lxx,
    const Eigen::Ref<const Eigen::MatrixXd>& kxxstar) {
  return (kxstarxstar - kxxstar.transpose() * lxx.solve(kxxstar))
      .selfadjointView<Eigen::Lower>()
      .llt()
      .matrixL();
}

/* Conditional marginal variance */
Eigen::VectorXd conditionalVar(const double kxstarxstar_diag,
                               const Eigen::LLT<Eigen::MatrixXd>& lxx,
                               const Eigen::Ref<const Eigen::MatrixXd>& kxxstar,
                               const double jitter) {
  return (kxstarxstar_diag -
          ((kxxstar.transpose() * lxx.solve(kxxstar)).diagonal()).array())
      .cwiseMax(jitter)
      .matrix();
}

/* Conditional marginal variance for 1 point only */
Eigen::VectorXd conditionalVar1d(
    const double kxstarxstar_diag, const Eigen::LLT<Eigen::MatrixXd>& lxx,
    const Eigen::Ref<const Eigen::RowVectorXd>& kxstarx, const double delta) {
  return (kxstarxstar_diag -
          ((kxstarx * lxx.solve(kxstarx.transpose())).diagonal()).array())
      .cwiseMax(delta)
      .matrix();
}

/* Conditional marginal standard deviation */
Eigen::VectorXd conditionalSd(const double kxstarxstar_diag,
                              const Eigen::LLT<Eigen::MatrixXd>& lxx,
                              const Eigen::Ref<const Eigen::MatrixXd>& kxxstar,
                              const double delta) {
  return (kxstarxstar_diag -
          ((kxxstar.transpose() * lxx.solve(kxxstar)).diagonal()).array())
      .cwiseMax(delta)
      .sqrt()
      .matrix();
}

// Outer addition of a vector with itself
Eigen::MatrixXd plusSelfOuter(const Eigen::VectorXd& x) {
  return (x.rowwise().replicate(x.size()) +
          x.transpose().colwise().replicate(x.size()))
      .selfadjointView<Eigen::Lower>();
}

// Outer subtraction of a vector with itself
Eigen::MatrixXd subtractSelfOuter(const Eigen::VectorXd& x) {
  return (x.rowwise().replicate(x.size()) -
          x.transpose().colwise().replicate(x.size()))
      .selfadjointView<Eigen::Lower>();
}

// Outer addition of two vectors
Eigen::MatrixXd plusOuter(const Eigen::VectorXd& x1,
                          const Eigen::VectorXd& x2) {
  return x1.rowwise().replicate(x2.size()) +
         x2.transpose().colwise().replicate(x1.size());
}

// Outer subtraction of two vectors
Eigen::MatrixXd subtractOuter(const Eigen::VectorXd& x1,
                              const Eigen::VectorXd& x2) {
  return x1.rowwise().replicate(x2.size()) -
         x2.transpose().colwise().replicate(x1.size());
}

// Vectorised scaled Euclidean distance of x with itself
Eigen::MatrixXd scaledDistanceSelf(const Eigen::MatrixXd& x,
                                   const Eigen::VectorXd& ls_sq) {
  Eigen::MatrixXd a = x.array().rowwise() / ls_sq.transpose().array().sqrt();
  Eigen::MatrixXd tau = plusSelfOuter(a.array().square().rowwise().sum());
  tau.noalias() += -2.0 * a * a.transpose();
  return tau;
}

// Vectorised scaled Euclidean distance between x1 and x2
Eigen::MatrixXd scaledDistance(const Eigen::MatrixXd& x1,
                               const Eigen::MatrixXd& x2,
                               const Eigen::VectorXd& ls_sq) {
  Eigen::MatrixXd a = x1.array().rowwise() / ls_sq.transpose().array().sqrt();
  Eigen::MatrixXd b = x2.array().rowwise() / ls_sq.transpose().array().sqrt();
  Eigen::MatrixXd tau = plusOuter(a.array().square().rowwise().sum(),
                                  b.array().square().rowwise().sum());
  tau.noalias() += -2.0 * a * b.transpose();
  return tau;
}

// Identity matrix multiplied by (sigma_sq + jitter)
Eigen::MatrixXd sigmaToMat(const int n, const double sigma_sq,
                           const double jitter) {
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n, n);
  return identity.array() * (sigma_sq + jitter);
}

// Helper function for 1D scaled distance
Eigen::VectorXd scaledDistance1dInner_(const Eigen::VectorXd& a,
                                       const Eigen::VectorXd& b,
                                       const Eigen::VectorXd& c) {
  return (a.array() - b.array()) / c.array().sqrt();
}

// Scaled Euclidean distance between vector xstar and rows of matrix x
Eigen::VectorXd scaledDistance1d(const Eigen::VectorXd& xstar,
                                 const Eigen::MatrixXd& x,
                                 const Eigen::VectorXd& ls_sq) {
  int n = x.rows();
  Eigen::VectorXd tau(n);
  for (int i = 0; i < n; i++) {
    tau(i) = scaledDistance1dInner_(xstar, x.row(i), ls_sq).squaredNorm();
  }
  return tau;
};

// (xstar(j) - x(:,j)) / ls_sq(j)
Eigen::VectorXd xstarMinusxOverls_(const double x_j,
                                   const Eigen::VectorXd& xcol,
                                   const double ls_sq_j) {
  return (x_j - xcol.array()) / ls_sq_j;
}

// Lower Cholesky Factor
Eigen::LLT<Eigen::MatrixXd> chol_(const Eigen::MatrixXd& x) {
  return x.selfadjointView<Eigen::Lower>().llt();
}

// ( ystar - mean) / sigma
Eigen::VectorXd calcGamma_(const Eigen::VectorXd& mean,
                           const Eigen::VectorXd& sd, const double ystar) {
  return (ystar - mean.array()) / sd.array();
}

// Helper function for expected improvement
Eigen::VectorXd expectedImprovementInner_(const Eigen::VectorXd& mean,
                                          const Eigen::VectorXd& sd,
                                          const double ystar) {
  return (ystar - mean.array()) * ncdf(calcGamma_(mean, sd, ystar)).array() +
         sd.array() * npdf(calcGamma_(mean, sd, ystar)).array();
}

// Helper function for upper confidence bound
Eigen::VectorXd upperConfidenceBoundInner_(const Eigen::VectorXd& mean,
                                           const Eigen::VectorXd& sd,
                                           const double sr_beta) {
  return mean.array() - sr_beta * sd.array();
}

// -0.5 Tr(a * dk)
double loglikDeriv_(const Eigen::MatrixXd &a, const Eigen::MatrixXd &dk) {
  return -0.5 * (a * dk).trace();
}

#endif  // __MISC_H_
