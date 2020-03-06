#define _USE_MATH_DEFINES

#include <Eigen/LU>
#include <Eigen/Dense>
#include <math.h>
#include <unsupported/Eigen/SpecialFunctions>

/* Vectorised standard normal distribution probability density function */
Eigen::VectorXd npdf(Eigen::Ref<const Eigen::VectorXd> x){
  return (1 / sqrt(2 * M_PI)) * (-(x.array().square() / 2)).exp();
}

/* Vectorised standard normal distribution cumulative density function */
Eigen::VectorXd ncdf(Eigen::Ref<const Eigen::VectorXd> x){
  return 0.5 * (1 + erf(x.array() / (sqrt(2))));
}

/* Standard normal distribution probability density function */
double npdfd(double x){
  return (1 / sqrt(2 * M_PI)) * exp(-(pow(x,2) / 2));
}

/* Standard normal distribution cumulative density function */
double ncdfd(double x){
  return 0.5 * (1 + erf(x / (sqrt(2))));
}

/* Derivative of standard normal distribution probability density function */
double npdfdg(double x){
  return -(x * exp(-((pow(x,2))/2))) / sqrt(2 * M_PI);
}

/*
 ----------------------------------------------------------- Squared Exponential
*/

Eigen::LLT<Eigen::MatrixXd> L_cov_se(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double alpha,
    Eigen::Ref<const Eigen::VectorXd> rho,
    const double sigma)
{
    Eigen::MatrixXd c(x.rows(), x.rows());
    double acc;
    for(int i = 0; i < x.rows(); i++){
        for(int j = i; j < x.rows(); j++){
          if (i!=j){
            acc = 0.0;
            for(int d = 0; d < x.cols(); d++){
              acc += pow(x(j, d) - x(i, d), 2.0) / rho(d);
            }
            c(j, i) = alpha * exp(-0.5 * acc);
          }
          if (i==j){
            c(i,i) = alpha + sigma;
          }
        }
      }
    return c.selfadjointView<Eigen::Lower>().llt();
}

Eigen::MatrixXd cov_se(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double alpha,
    Eigen::Ref<const Eigen::VectorXd> rho,
    const double sigma)
{
    Eigen::MatrixXd c(x.rows(), x.rows());
    double acc;
    for(int i = 0; i < x.rows(); i++){
        for(int j = i; j < x.rows(); j++){
          if (i!=j){
            acc = 0.0;
            for(int d = 0; d < x.cols(); d++){
              acc += pow(x(j, d) - x(i, d), 2.0) / rho(d);
            }
            c(j, i) = alpha * exp(-0.5 * acc);
          }
          if (i==j){
            c(i,i) = alpha + sigma;
          }
        }
      }
    return c.selfadjointView<Eigen::Lower>();
}

Eigen::MatrixXd cross_cov_se(
    Eigen::Ref<const Eigen::MatrixXd> x1,
    Eigen::Ref<const Eigen::MatrixXd> x2,
    double alpha,
    Eigen::Ref<const Eigen::VectorXd> rho)
{
    Eigen::RowVectorXd rho_sqr = rho.transpose().array().sqrt();
    Eigen::MatrixXd c(x1.rows(), x2.rows());
    Eigen::MatrixXd al = x1.array().rowwise() / rho_sqr.array();
    Eigen::VectorXd tmp1 = al.array().square().rowwise().sum();
    Eigen::MatrixXd bl = x2.array().rowwise() / rho_sqr.array();
    Eigen::VectorXd tmp2 = bl.array().square().rowwise().sum();
    c = tmp1.rowwise().replicate(tmp2.size()) + tmp2.transpose().colwise().replicate(tmp1.size());
    c.noalias() += -2.0 * al * bl.transpose();
    return alpha * ((-0.5 * c).array().exp()).matrix();
}

Eigen::RowVectorXd cov_1d_se(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double alpha,
    Eigen::Ref<const Eigen::VectorXd> rho)
{
    Eigen::RowVectorXd c(x.rows());
    double acc;
    for(int i = 0; i < x.rows(); i++){
        acc = 0;
        for(int d = 0; d < x.cols(); d++){
            acc += pow(xstar1d(d) - x(i,d), 2.0) / rho(d);
        }
        c(i) = alpha * exp(-0.5 * acc);
    }
    return c;
}

Eigen::RowVectorXd cross_cov_grad_se(
    Eigen::Ref<const Eigen::RowVectorXd> xstar,
    Eigen::Ref<const Eigen::MatrixXd> x,
    int d,
    double asq,
    Eigen::Ref<const Eigen::VectorXd> lsq,
    Eigen::Ref<const Eigen::MatrixXd> kxstarx)
{
    return ((xstar(d) - x.col(d).array()) / lsq(d)).transpose().array() * kxstarx.array();
}

/*
----------------------------------------------------------------------- Matern52
*/

Eigen::LLT<Eigen::MatrixXd> L_cov_m52(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double alpha,
    Eigen::Ref<const Eigen::VectorXd> rho,
    const double sigma)
{
    Eigen::MatrixXd c(x.rows(), x.rows());
    double acc;
    double r;
    double sr5 = sqrt(5);
    double rat = 5.0 / 3.0;
    for(int i = 0; i < x.rows(); i++){
        for(int j = i; j < x.rows(); j++){
            if (i!=j){
                acc = 0.0;
                for(int d = 0; d < x.cols(); d++){
                    acc += pow(x(j, d) - x(i, d), 2.0) / rho(d);
                }
                r = sqrt(acc);
                c(j, i) = alpha * (1 + sr5*r + rat*acc) * exp(-sr5*r);
            }
            if (i==j){
                c(i,i) = alpha + sigma;
            }
        }
    }
    return c.selfadjointView<Eigen::Lower>().llt();
}

Eigen::MatrixXd cov_m52(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double alpha,
    Eigen::Ref<const Eigen::VectorXd> rho,
    const double sigma)
{
    Eigen::MatrixXd c(x.rows(), x.rows());
    double acc;
    double r;
    double sr5 = sqrt(5);
    double rat = 5.0 / 3.0;
    for(int i = 0; i < x.rows(); i++){
        for(int j = i; j < x.rows(); j++){
            if (i!=j){
                acc = 0.0;
                for(int d = 0; d < x.cols(); d++){
                    acc += pow(x(j, d) - x(i, d), 2.0) / rho(d);
                }
                r = sqrt(acc);
                c(j, i) = alpha * (1 + sr5*r + rat*acc) * exp(-sr5*r);
            }
            if (i==j){
                c(i,i) = alpha + sigma;
            }
        }
    }
    return c.selfadjointView<Eigen::Lower>();
}

Eigen::MatrixXd cross_cov_m52(
    Eigen::Ref<const Eigen::MatrixXd> x1,
    Eigen::Ref<const Eigen::MatrixXd> x2,
    double alpha,
    Eigen::Ref<const Eigen::VectorXd> rho)
{
    double sr5 = sqrt(5);
    double rat = 5.0 / 3.0;
    Eigen::RowVectorXd rho_sqr = rho.transpose().array().sqrt();
    Eigen::MatrixXd r(x1.rows(), x2.rows());
    Eigen::MatrixXd r_sqr(x1.rows(), x2.rows());
    Eigen::MatrixXd al = x1.array().rowwise() / rho_sqr.array();
    Eigen::VectorXd tmp1 = al.array().square().rowwise().sum();
    Eigen::MatrixXd bl = x2.array().rowwise() / rho_sqr.array();
    Eigen::VectorXd tmp2 = bl.array().square().rowwise().sum();
    r = tmp1.rowwise().replicate(tmp2.size()) + tmp2.transpose().colwise().replicate(tmp1.size());
    r.noalias() += -2.0 * al * bl.transpose();
    r_sqr = r.array().sqrt();
    return alpha * (1 + sr5 * r_sqr.array() + rat * r.array()) * (-sr5 * r_sqr).array().exp();
}

Eigen::RowVectorXd cov_1d_m52(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double alpha,
    Eigen::Ref<const Eigen::VectorXd> rho)
{
    Eigen::RowVectorXd c(x.rows());
    double acc;
    double r;
    double sr5 = sqrt(5.0);
    double rat = 5.0 / 3.0;
    for(int i = 0; i < x.rows(); i++){
        acc = 0;
        for(int d = 0; d < x.cols(); d++){
            acc += pow(xstar1d(d) - x(i,d), 2.0) / rho(d);
        }
        r = sqrt(acc);
        c(i) = alpha * (1 + sr5 * r + rat*acc) * exp(-sr5*r);
    }
    return c;
}

Eigen::RowVectorXd cross_cov_grad_m52(
                                     Eigen::Ref<const Eigen::RowVectorXd> xstar1d,
                                     Eigen::Ref<const Eigen::MatrixXd> x,
                                     int d,
                                     double alpha,
                                     Eigen::Ref<const Eigen::VectorXd> rho)
{
    Eigen::RowVectorXd dc(x.rows());
    double r;
    double sr5 = sqrt(5);
    double tmp_1;
    double tmp_2;
    double rat_10_3 = 10.0 /3.0;
    double rat_5_3 = 5.0 / 3.0;

    for(int i = 0; i < x.rows(); i++){
      r = 0;
      for(int d_internal = 0; d_internal < x.cols(); d_internal++){
        r += pow(xstar1d(d_internal) - x(i,d_internal), 2.0) / rho(d_internal);
      }
      tmp_1 = sqrt(r) * sr5;
      tmp_2 = alpha * (rat_10_3 - 5.0 - rat_5_3 * tmp_1) * exp(-tmp_1);
      dc(i) = -((xstar1d(d) - x(i,d)) / rho(d)) * tmp_2;
    }
    return dc;
}
