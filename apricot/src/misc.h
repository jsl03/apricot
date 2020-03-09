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
