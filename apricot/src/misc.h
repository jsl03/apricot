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

/* Conditional covariance matrix */
Eigen::MatrixXd conditionalCov(
    const Eigen::Ref<const Eigen::MatrixXd>& kxstarxstar,
    const Eigen::LLT<Eigen::MatrixXd> &lxx,
    const Eigen::Ref<const Eigen::MatrixXd>& kxxstar
    )
{
     return (
          kxstarxstar - kxxstar.transpose() * lxx.solve(kxxstar)
          ).selfadjointView<Eigen::Lower>();
}

/* Cholesky factor of conditional covariance matrix */
Eigen::MatrixXd conditionalCovChol(
    const Eigen::Ref<const Eigen::MatrixXd>& kxstarxstar,
    const Eigen::LLT<Eigen::MatrixXd> &lxx,
    const Eigen::Ref<const Eigen::MatrixXd>& kxxstar
    )
{
     return (
          kxstarxstar - kxxstar.transpose() * lxx.solve(kxxstar)
          ).selfadjointView<Eigen::Lower>().llt().matrixL();
}

/* Conditional marginal variance */
Eigen::VectorXd conditionalVar(
     const double kxstarxstar_diag,
     const Eigen::LLT<Eigen::MatrixXd> &lxx,
     const Eigen::Ref<const Eigen::MatrixXd>& kxxstar,
     const double delta
     )
{
     return (
          kxstarxstar_diag -
          ((kxxstar.transpose() * lxx.solve(kxxstar)).diagonal()).array()
          ).cwiseMax(delta).matrix();
}

/* Conditional marginal variance for 1 point only */
Eigen::VectorXd conditionalVar1d(
     const double kxstarxstar_diag,
     const Eigen::LLT<Eigen::MatrixXd> &lxx,
     const Eigen::Ref<const Eigen::RowVectorXd>& kxstarx,
     const double delta
     )
{
     return (
          kxstarxstar_diag -
          ((kxstarx * lxx.solve(kxstarx.transpose())).diagonal()).array()
          ).cwiseMax(delta).matrix();
}

/* Conditional marginal standard deviation */
Eigen::VectorXd conditionalSd(
     const double kxstarxstar_diag,
     const Eigen::LLT<Eigen::MatrixXd> &lxx,
     const Eigen::Ref<const Eigen::MatrixXd>& kxxstar,
     const double delta
     )
{
     return (
          kxstarxstar_diag -
          ((kxxstar.transpose() * lxx.solve(kxxstar)).diagonal()).array()
          ).cwiseMax(delta).sqrt().matrix();
}
