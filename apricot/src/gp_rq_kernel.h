// This file is licensed under Version 3.0 of the GNU General Public
// License. See LICENSE for a text of the license.
// -----------------------------------------------------------------------------

// Lower cholesky factor of RQ kernel sample covariance matrix
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
//
// Outputs:
//     lxx : (n, n) array
//         lower Cholesky factor of cov(x, x | amp_sq, ls_sq, kappa sigma_sq)
//
Eigen::LLT<Eigen::MatrixXd> L_cov_rq(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    const double kappa,
    const double sigma_sq)
{
    Eigen::MatrixXd c(x.rows(), x.rows());
    double tau;
    for(int i = 0; i < x.rows(); i++){
        for(int j = i; j < x.rows(); j++){
            if (i!=j){
                tau = 0.0;
                for(int d = 0; d < x.cols(); d++){
                    tau += pow(x(j, d) - x(i, d), 2.0) / ls_sq(d);
                }
                c(j, i) = amp_sq * pow(1.0 + 0.5 * tau / kappa, -kappa); 
            }
            if (i==j){
                c(i,i) = amp_sq + sigma_sq;
            }
        }
    }
    return c.selfadjointView<Eigen::Lower>().llt();
}

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
//
// Outputs:
//     kxx : (n, n) array
//         Covariance matrix cov(x, x | amp_sq, ls_sq, kappa, sigma_sq)
//
Eigen::MatrixXd cov_rq(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    const double kappa,
    const double sigma_sq)
{
    Eigen::MatrixXd c(x.rows(), x.rows());
    double tau;
    for(int i = 0; i < x.rows(); i++){
        for(int j = i; j < x.rows(); j++){
            if (i!=j){
                tau = 0.0;
                for(int d = 0; d < x.cols(); d++){
                    tau += pow(x(j, d) - x(i, d), 2.0) / ls_sq(d);
                }
                c(j, i) = amp_sq * pow(1.0 + 0.5 * tau / kappa, -kappa);
            }
            if (i==j){
                c(i,i) = amp_sq + sigma_sq;
            }
        }
    }
    return c.selfadjointView<Eigen::Lower>();
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
//     kappa : double
//         relative scale weighting hyperparameter
//
// Outputs:
//     kx1x2 : (n, m) array
//         Cross covariance matrix k(x1, x2 | amp_sq, ls_sq, kappa)
//
// Notes:
//     This "vectorised" version of the calculation is faster than the explicit
//     loop.
//
Eigen::MatrixXd cross_cov_rq(
    Eigen::Ref<const Eigen::MatrixXd> x1,
    Eigen::Ref<const Eigen::MatrixXd> x2,
    double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    double kappa)
{
    Eigen::RowVectorXd ls = ls_sq.transpose().array().sqrt();
    Eigen::MatrixXd tau(x1.rows(), x2.rows());
    Eigen::MatrixXd tmp(x1.rows(), x2.rows());
    Eigen::MatrixXd al = x1.array().rowwise() / ls.array();
    Eigen::VectorXd tmp1 = al.array().square().rowwise().sum();
    Eigen::MatrixXd bl = x2.array().rowwise() / ls.array();
    Eigen::VectorXd tmp2 = bl.array().square().rowwise().sum();
    tau = tmp1.rowwise().replicate(tmp2.size()) +
        tmp2.transpose().colwise().replicate(tmp1.size());
    tau.noalias() += -2.0 * al * bl.transpose();
    return amp_sq * pow(1.0 + 0.5 * tau.array() / kappa, -kappa);
}

// RQ kernel 1d cross covariance calculation
//
// Inputs:
//     just_xstar : (d) array
//         a single d dimensional point
//     x : (n, d) array
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//     kappa : double
//         relative scale weighting hyperparameter
//
// Outputs:
//     kxstarx : (n) array
//         covariance between just_xstar and x, i.e.
//         kxstarx = k(just_xstar, x | amp_sq, ls_sq)
//
Eigen::RowVectorXd cov_1d_rq(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    const double kappa)
{
    Eigen::RowVectorXd c(x.rows());
    double tau;
    for(int i = 0; i < x.rows(); i++){
        tau = 0;
        for(int d = 0; d < x.cols(); d++){
            tau += pow(xstar1d(d) - x(i,d), 2.0) / ls_sq(d);
        }
        c(i) = amp_sq * pow(1.0 + 0.5 * tau / kappa, -kappa);
    }
    return c;
}

// RQ kernel 1d cross covariance partial derivatives
//
// Inputs:
//     just_xstar : (d) array
//         a single d dimensional point
//     x : (n, d) array
//         n points in d dimensions
//     d : int
//         dimension in which to compute the partial derivative
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//     kappa : double
//         relative scale weighting hyperparameter
//
// Outputs:
//     dkxstarx_dxstar : (n) array
//         partial derivatives of k(just_xstar, x | amp_sq, ls_sq, kappa)
//         with respect to just_xstar in dimension d
//
Eigen::RowVectorXd cross_cov_grad_rq(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,
    Eigen::Ref<const Eigen::MatrixXd> x,
    int d,
    double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    double kappa)
{
    Eigen::RowVectorXd dc(x.rows());
    double tau;
    double tmp;
    for(int i = 0; i < x.rows(); i++){
        tau = 0;
        for(int d_internal = 0; d_internal < x.cols(); d_internal++){
            tau += pow(xstar1d(d_internal) - x(i,d_internal), 2.0)
                / ls_sq(d_internal);
        }
        tmp = -amp_sq * pow(1.0 + 0.5 * tau / kappa, -kappa - 1.0);
        dc(i) = -((xstar1d(d) - x(i,d)) / ls_sq(d)) * tmp;
    }
    return dc;
}

// Class for a Gaussian Process with an EQ covariance kernel
//
// Private Attributes:
//     x : (n, d) array
//         n points in d dimensions
//     y : (n)
//         n scalar responses
//     amp_sq : (s) array
//         a total of t squared marginal standard deviation (marginal variance)
//         hyperparameter samples
//     ls_sq : (s, d) array
//         a total of t length d vectors representing dimensionwise
//         (anisotropic) squared lengthscale hyperparameter samples
//     sigma_sq : (s) array
//         a total of s squared Gaussian noise standard deviation
//         (noise variance) hyperparameter samples
//     ks : (s) array
//         a total of s relative scale weighting hyperparameter samples
//     delta : double
//         stability jitter ("nugget")
//     lxxs : List of size s
//         A total of s memoized (n, n) sample covariance cholesky factors. The
//         kth element of lxxs corresponds to the lower Cholesky factor of the
//         sample covariance matrix obtained by
//         k(x, x | amp_sq_k, ls_sq_k, sigma_sq_k), that is, the kth set of
//         hyperparameter samples.
//     axxs : List of size s
//         A total of s memoized (n) dot products between the inverse sample
//         covariance matrix and the sample responses. The kth element of axxs
//         corresponds is equivalent to lxxs[k].solve(y).
//
// Public Attributes:
//     xdata : (n,d) array
//         n points in d dimensions
//     ydata : (n) array
//         n scalar responses
//     amp : (s) array
//         a total of s marginal standard deviation hyperparameter samples
//     ls : (s, d) array
//         a total of s length d vectors representing dimensionwise
//         (anisotropic) lengthscale samples
//     kappa : (s) array
//         a total of s relative scale weighting hyperparameters
//     sigma : (s) array
//         a total of s Gaussian noise standard deviation hyperparameter
//         samples
//     jitter : double
//         stability jitter
//
// Public Methods:
//     view_x
//         view the data x
//     view_y
//         view the data y
//     view_parameters
//         view the hyperparameters (amp_sq, ls_sq, sigma_sq)
//     view_lxx
//         view lower Cholesky factors of k(x,x)
//     posterior
//         posterior joint distribution
//     posterior_chol
//         posterior joint distribution (Cholesky parametrisation)
//     marginals
//         posterior marginals
//     E
//         posterior expectation
//     px
//         pure exploration (negative posterior variance)
//     px_jac
//         pure exploration and its jacobian
//     ei
//         expected improvement
//     ei_jac
//         expected improvement and its jacobian
//     ucb
//         upper confidence bound
//     ucb_jac
//         upper confidence bound and its jacobian
//     loo_cv
//         leave one out cross validation scores for sample data
//     entropy
//         differential entropy
//
class GpRqKernel
{
private:
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
    Eigen::VectorXd amp_sq;
    Eigen::MatrixXd ls_sq;
    Eigen::VectorXd ks;
    Eigen::VectorXd sigma_sq;
    double delta;
    std::vector<Eigen::LLT<Eigen::MatrixXd>> lxxs;
    std::vector<Eigen::VectorXd> axxs;

public:
    GpRqKernel(
        const Eigen::MatrixXd &xdata,
        const Eigen::VectorXd &ydata,
        const Eigen::VectorXd &amp,
        const Eigen::MatrixXd &ls,
        const Eigen::VectorXd &kappa,
        const Eigen::VectorXd &sigma,
        const double jitter);

    const Eigen::MatrixXd
    view_x();

    const Eigen::VectorXd
    view_y();

    std::tuple<
        const Eigen::VectorXd,
        const Eigen::MatrixXd,
        const Eigen::VectorXd,
        const Eigen::VectorXd
        >
    view_parameters();

    std::vector<Eigen::MatrixXd>
    view_lxx();

    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>>
    posterior(
        Eigen::Ref<const Eigen::MatrixXd> xstar
        );

    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>>
    posterior_chol(
        Eigen::Ref<const Eigen::MatrixXd> xstar
        );

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
    marginals(
        Eigen::Ref<const Eigen::MatrixXd> xstar
        );

    Eigen::VectorXd
    E(
        Eigen::Ref<const Eigen::MatrixXd> xstar
        );

    std::tuple<double, Eigen::VectorXd>
    E_jac(
        Eigen::Ref<const Eigen::RowVectorXd> xstar
        );

    Eigen::VectorXd
    px(
        Eigen::Ref<const Eigen::MatrixXd> xstar
        );

    std::tuple<Eigen::VectorXd, Eigen::VectorXd>
    px_jac(
        Eigen::Ref<const Eigen::RowVectorXd> xstar
        );

    Eigen::VectorXd
    ei(
        Eigen::Ref<const Eigen::MatrixXd> xstar
        );

    std::tuple<double, Eigen::VectorXd>
    ei_jac(
        Eigen::Ref<const Eigen::RowVectorXd> xstar
        );

    Eigen::VectorXd
    ucb(
        Eigen::Ref<const Eigen::MatrixXd> xstar, double beta
        );

    std::tuple<double, Eigen::VectorXd>
    ucb_jac(
        Eigen::Ref<const Eigen::RowVectorXd> xstar, double beta
        );

    Eigen::MatrixXd
    loo_cv();

    double
    entropy(
        Eigen::Ref<const Eigen::MatrixXd> xstar
        );
};

// class-constructor
GpRqKernel::GpRqKernel(
    const Eigen::MatrixXd &xdata,
    const Eigen::VectorXd &ydata,
    const Eigen::VectorXd &amp,
    const Eigen::MatrixXd &ls,
    const Eigen::VectorXd &kappa,
    const Eigen::VectorXd &sigma,
    const double jitter):
        x(xdata),
        y(ydata),
        amp_sq(amp.array().square()),
        ls_sq(ls.array().square()),
        ks(kappa),
        sigma_sq(sigma.array().square()),
        delta(jitter)
{
    for (int k=0; k < amp_sq.size(); k++){
        lxxs.reserve(amp_sq.size());
        axxs.reserve(amp_sq.size());
        lxxs.push_back(
            L_cov_rq(x, amp_sq(k), ls_sq.row(k), ks(k), sigma_sq(k))
            );
        axxs.push_back(lxxs[k].solve(y));
    }
}

// view the data x
//
// Outputs:
//     x : (n, d)
//         n points in d dimensions
//
const Eigen::MatrixXd
GpRqKernel::view_x()
{
    return x;
}

// view the data y
//
// Outputs:
//     y : (n) array
//         n scalar responses
//
const Eigen::VectorXd
GpRqKernel::view_y()
{
    return y;
}

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
std::tuple<
    const Eigen::VectorXd,
    const Eigen::MatrixXd,
    const Eigen::VectorXd,
    const Eigen::VectorXd
    >
GpRqKernel::view_parameters()
{
    return std::make_tuple(amp_sq, ls_sq, ks, sigma_sq);
}

// view the lower Cholesky factors of k(x, x | amp, ls, sigma)
//
// Outputs:
//     lxxs : list of (n,n) array
//         Length s list of the lower Cholesky factors of the sample covariance
//         matrices computed using the kth set of hyperparameters, that is:
//         lxxs(k) = chol(k(x, x | amp_k, ls_k, sigma_k))
//
std::vector<Eigen::MatrixXd> GpRqKernel::view_lxx()
{
    std::vector<Eigen::MatrixXd> lxx_views;
    lxx_views.reserve(amp_sq.size());
    for (int k=0; k < amp_sq.size(); k++){
        lxx_views.push_back(lxxs[k].matrixL());
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
GpRqKernel::posterior(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::MatrixXd means(xstar.rows(), amp_sq.size());
    std::vector<Eigen::MatrixXd> cov_mats;
    cov_mats.reserve(amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
         kxxstar = cross_cov_rq(x, xstar, amp_sq(k), ls_sq.row(k), ks(k));
         means.col(k) = kxxstar.transpose() * axxs[k];
         cov_mats.push_back(
              conditionalCov(
                   cov_rq(xstar, amp_sq(k), ls_sq.row(k), ks(k), sigma_sq(k)),
                   lxxs[k],
                   kxxstar)
              );
    }
    return std::make_tuple(means, cov_mats);
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
GpRqKernel::posterior_chol(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::MatrixXd means(xstar.rows(), amp_sq.size());
    std::vector<Eigen::MatrixXd> cov_mat_chols;
    cov_mat_chols.reserve(amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k < amp_sq.size(); k++){
        kxxstar = cross_cov_rq(x, xstar, amp_sq(k), ls_sq.row(k), ks(k));
        means.col(k) = kxxstar.transpose() * axxs[k];
        cov_mat_chols.push_back(
              conditionalCovChol(
                   cov_rq(xstar, amp_sq(k), ls_sq.row(k), ks(k), sigma_sq(k)),
                   lxxs[k],
                   kxxstar)
              );
    }
    return std::make_tuple(means, cov_mat_chols);
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
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
GpRqKernel::marginals(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::MatrixXd means(xstar.rows(), amp_sq.size());
    Eigen::MatrixXd vars(xstar.rows(), amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k < amp_sq.size(); k++){
        kxxstar = cross_cov_rq(x, xstar, amp_sq(k), ls_sq.row(k), ks(k));
        means.col(k) = kxxstar.transpose() * axxs[k];
        vars.col(k) = conditionalVar(
            amp_sq(k) + sigma_sq(k),
            lxxs[k],
            kxxstar,
            delta
            );
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
Eigen::VectorXd
GpRqKernel::E(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::VectorXd expectation = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_rq(x, xstar, amp_sq(k), ls_sq.row(k), ks(k));
        expectation.noalias() += (kxxstar.transpose() * axxs[k]);
    }
    return expectation / amp_sq.size();
}

// posterior expectation and jacobian
//
// Inputs:
//     just_xstar : (d) array
//         a single d-dimensional point at which the posterior expectation
//         and its partial derivatives with respect to just_xstar should be
//         computed
//
// Outputs:
//     expectation : double
//         the posterior expectation at just_xstar, integrated over the s
//         hyperparameter samples.
//     expectation_jac : (d) array
//         the partial derivatives of the posterior expectation at just_xstar
//         with respect to just_xstar
//
std::tuple<double, Eigen::VectorXd>
GpRqKernel::E_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar
    )
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    double expectation = 0;
    Eigen::VectorXd expectation_jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_eq(xstar, x, amp_sq(k), ls_sq.row(k));
        expectation += kxstarx.dot(axxs[k]);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_rq(
                 xstar,
                 x,
                 d,
                 amp_sq(k),
                 ls_sq.row(k),
                 ks(k));
            expectation_jac(d) -= dkxstarx_dxstar.dot(axxs[k]);
        }
    }
    return std::make_tuple(
         expectation / amp_sq.size(),
         expectation_jac / amp_sq.size());
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
Eigen::VectorXd
GpRqKernel::px(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::VectorXd px = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k < amp_sq.size(); k++){
        px.noalias() -= conditionalVar(
             amp_sq(k) + sigma_sq(k),
             lxxs[k],
             cross_cov_rq(x, xstar, amp_sq(k), ls_sq.row(k), ks(k)),
             delta
             );
    }
    return px / amp_sq.size();
}

// pure exploration and jacobian
//
// Inputs:
//     just_xstar : (d) array
//         a single d-dimensional point at which the pure exploration
//         acquisition function and its partial derivatives with respect
//         to just_xstar should be computed
//
// Outputs:
//     px : double
//         the pure exploration at just_xstar, integrated over the s
//         hyperparameter samples.
//     px_jac : (d) array
//         the partial derivatives of the pure exploration at just_xstar
//         with respect to just_xstar
//
// Notes:
//     px is negative and equivalent to the (negative) expectation of the
//     marginal variance
//
std::tuple<Eigen::VectorXd, Eigen::VectorXd>
GpRqKernel::px_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar
    )
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    Eigen::RowVectorXd px = Eigen::VectorXd::Zero(1);
    Eigen::MatrixXd px_jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k < amp_sq.size(); k++){
        kxstarx = cov_1d_eq(xstar, x, amp_sq(k), ls_sq.row(k));
        px.noalias() -= conditionalVar1d(
             amp_sq(k) + sigma_sq(k),
             lxxs[k],
             kxstarx,
             delta
             );
        for (int d=0; d < xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_rq(
                xstar,
                x,
                d,
                amp_sq(k),
                ls_sq.row(k),
                ks(k));
            px_jac(d) += kxstarx * lxxs[k].solve(-dkxstarx_dxstar.transpose());
        }
    }
    return std::make_tuple(
         px / amp_sq.size(),
         px_jac*2 / amp_sq.size());
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
Eigen::VectorXd
GpRqKernel::ei(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ei = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::VectorXd gamma(xstar.rows());
    double ystar = y.minCoeff();
    for (int k=0; k < amp_sq.size(); k++){
        kxxstar = cross_cov_rq(x, xstar, amp_sq(k), ls_sq.row(k), ks(k));
        mean = kxxstar.transpose() * axxs[k];
        sd = conditionalSd(
             amp_sq(k) + sigma_sq(k),
             lxxs[k],
             kxxstar,
             delta
             );
        gamma = (ystar - mean.array()).array() / sd.array();
        ei.array() -= (
             (ystar - mean.array()) * ncdf(gamma).array() +
             sd.array() * npdf(gamma).array()
             ).array();
    }
    return ei / amp_sq.size();
}

// expected improvement (EI) and jacobian
//
// Inputs:
//     just_xstar : (d) array
//         a single d-dimensional point at which the expected improvement
//         acquisition function and its partial derivatives with respect
//         to just_xstar should be computed
//
// Outputs:
//     expected_improvement : double
//         the expected improvement at just_xstar, integrated over the s
//         hyperparameter samples.
//     expected_improvement_jac : (d) array
//         the partial derivatives of the expected improvement at just_xstar
//         with respect to just_xstar
//
// Notes:
//     This implementation of EI is negative and seeks the minimum, that is,
//     this function measures the expected reduction over the incumbent minimum.
//
std::tuple<double, Eigen::VectorXd>
GpRqKernel::ei_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar
    )
{
    double ei = 0;
    double mean;
    double mean_grad;
    double var;
    double sd;
    double sd_grad;
    double gamma;
    double gamma_grad;
    double ystar = y.minCoeff();
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    Eigen::VectorXd ei_jac(xstar.size());
    for (int k=0; k < amp_sq.size(); k++){
        kxstarx = cov_1d_eq(xstar, x, amp_sq(k), ls_sq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (amp_sq(k) + sigma_sq(k)) -
             kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        gamma = (ystar - mean) / sd;
        ei -= (ystar - mean) * ncdfd(gamma) + sd * npdfd(gamma);
        for (int d=0; d < xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_rq(
                xstar,
                x,
                d,
                amp_sq(k),
                ls_sq.row(k),
                ks(k));
            sd_grad = -(
                kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose()))
                ) / sd;
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            gamma_grad = (
                ((mean - ystar) * sd_grad) - (sd * mean_grad)
                ) / var;
            ei_jac(d) -=
                 (ystar-mean) *
                 gamma_grad * npdfd(gamma) -
                 ncdfd(gamma) * mean_grad +
                 npdfd(gamma) * sd_grad + sd * gamma_grad * npdfdg(gamma);
        }
    }
    return std::make_tuple(
         ei / amp_sq.size(),
         ei_jac / amp_sq.size());
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
Eigen::VectorXd
GpRqKernel::ucb(
    Eigen::Ref<const Eigen::MatrixXd> xstar,
    double beta
    )
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ucb = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k < amp_sq.size(); k++){
        kxxstar = cross_cov_rq(x, xstar, amp_sq(k), ls_sq.row(k), ks(k));
        mean = kxxstar.transpose() * axxs[k];
        ucb.array() +=
             mean.array() - (sqrt(beta) * conditionalSd(
                                  amp_sq(k) + sigma_sq(k),
                                  lxxs[k],
                                  kxxstar,
                                  delta
                                  )).array();
    }
    return ucb / amp_sq.size();
}

// expected improvement (EI) and jacobian
//
// Inputs:
//     just_xstar : (d) array
//         a single d-dimensional point at which the expected improvement
//         acquisition function and its partial derivatives with respect
//         to just_xstar should be computed
//     beta : double
//         Parameter controlling the trade between exploration and exploitation
//
// Outputs:
//     expected_improvement : double
//         the expected improvement at just_xstar, integrated over the s
//         hyperparameter samples.
//     expected_improvement_jac : (d) array
//         the partial derivatives of the expected improvement at just_xstar
//         with respect to just_xstar
//
// Notes:
//     This implementation of UCB is negative and seeks the minimum: strictly
//     speaking this function measures a (negative) lower confidence bound.
//
std::tuple<double, Eigen::VectorXd>
GpRqKernel::ucb_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar,
    double beta
    )
{
    double ucb = 0;
    double mean;
    double mean_grad;
    double var;
    double sd;
    double sd_grad;
    double sr_beta = sqrt(beta);
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    Eigen::VectorXd ucb_jac(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_eq(xstar, x, amp_sq(k), ls_sq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (amp_sq(k) + sigma_sq(k)) -
             kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        ucb += mean - (sr_beta * sd);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_rq(
                xstar,
                x,
                d,
                amp_sq(k),
                ls_sq.row(k),
                ks(k));
            sd_grad =
                 -(kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose())))
                 / sd;
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            ucb_jac(d) += mean_grad - (sr_beta * sd_grad);
        }
    }
    return std::make_tuple(
         ucb / amp_sq.size(),
         ucb_jac / amp_sq.size()
         );
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
Eigen::MatrixXd GpRqKernel::loo_cv()
{
    Eigen::MatrixXd cv(amp_sq.rows(), x.rows());
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x.rows(), x.rows());
    Eigen::VectorXd c_diag;
    for (int k=0; k<amp_sq.size(); k++){
        c_diag = lxxs[k].solve(I).diagonal();
        cv.row(k) = -0.5*log(2*M_PI) +
            (0.5*(c_diag.array().log())) -
            0.5 * (axxs[k].array().square() / c_diag.array());
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
double GpRqKernel::entropy(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    const double e = std::exp(1.0);
    double h = 0;
    double tmp = xstar.rows() / 2.0 *  log(e * M_PI * 2.0);
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::MatrixXd c(xstar.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_rq(x, xstar, amp_sq(k), ls_sq.row(k), ks(k));
        // computes
        //
        // h = tmp + 0.5 * 2 * log(det(Sigma))
        // = tmp + 0.5 * 2 * log(det(kzz - kzx * kxx_inv * kxz))
        // = tmp + sum(log(diag(chol(kzz - kzx * kxx_inv * kxz))))
        //
        // in a way that Eigen can (hopefully) optimise
        h += tmp + (
            cov_rq(xstar, amp_sq(k), ls_sq.row(k), ks(k), sigma_sq(k)).array() -
            ((kxxstar.transpose() * lxxs[k].solve(kxxstar))).array()
            ).matrix().llt().matrixLLT().diagonal().array().log().sum();
    }
    return h / amp_sq.size();
}
