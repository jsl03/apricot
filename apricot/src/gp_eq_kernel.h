/*
 *
 * License goes here
 *
 */

// Lower cholesky factor of EQ kernel sample covariance matrix
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
//     L : (n, n) array
//         lower Cholesky factor of cov(x, x | amp_sq, ls_sq, sigma_sq)
//
Eigen::LLT<Eigen::MatrixXd> L_cov_eq(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    const double sigma_sq
    )
{
    Eigen::MatrixXd c(x.rows(), x.rows());
    double tau = 0.0;
    for(int i = 0; i < x.rows(); i++){
        for(int j = i; j < x.rows(); j++){
            if (i!=j){
                tau = 0.0;
                for(int d = 0; d < x.cols(); d++){
                    tau += pow(x(j, d) - x(i, d), 2.0) / ls_sq(d);
                }
                c(j, i) = amp_sq * exp(-0.5 * tau);
            }
            if (i==j){
                c(i,i) = amp_sq + sigma_sq;
            }
        }
    }
    return c.selfadjointView<Eigen::Lower>().llt();
}

// Lower cholesky factor of EQ kernel sample covariance matrix
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
//     C_x1_x2 : (n, n) array
//         Covariance matrix cov(x, x | amp_sq, ls_sq, sigma_sq)
//
Eigen::MatrixXd cov_eq(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    const double sigma_sq
    )
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
                c(j, i) = amp_sq * exp(-0.5 * tau);
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
//
// Outputs:
//     C_x1_x2 : (n, m) array
//         Cross covariance matrix k(x1, x2 | amp_sq, ls_sq)
//
// Notes:
//     This "vectorised" version of the calculation is faster than the explicit
//     loop.
//
Eigen::MatrixXd cross_cov_eq(
    Eigen::Ref<const Eigen::MatrixXd> x1,
    Eigen::Ref<const Eigen::MatrixXd> x2,
    double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq
    )
{
    Eigen::RowVectorXd ls = ls_sq.transpose().array().sqrt();
    Eigen::MatrixXd tau(x1.rows(), x2.rows());
    Eigen::MatrixXd al = x1.array().rowwise() / ls.array();
    Eigen::VectorXd tmp1 = al.array().square().rowwise().sum();
    Eigen::MatrixXd bl = x2.array().rowwise() / ls.array();
    Eigen::VectorXd tmp2 = bl.array().square().rowwise().sum();
    // next is an outer addition of two vectors:
    // it is equivalent to numpy.add.outer(tmp1, tmp2)
    tau = tmp1.rowwise().replicate(tmp2.size()) +
        tmp2.transpose().colwise().replicate(tmp1.size());
    tau.noalias() += -2.0 * al * bl.transpose();
    return amp_sq * ((-0.5 * tau).array().exp()).matrix();
}

// EQ kernel 1d cross covariance calculation
//
// Inputs:
//     xstar1d : (d) array
//         a single d dimensional point
//     x : (n, d) array
//         n points in d dimensions
//     amp_sq : double
//         squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array
//         dimensionwise (anisotropic) squared lengthscales
//
// Outputs:
//     C_1d : (n) array
//         covariance between xstar1d and x, i.e. k(xstar1d, x | amp_sq, ls_sq)
//
Eigen::RowVectorXd cov_1d_eq(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq
    )
{
    Eigen::RowVectorXd c(x.rows());
    double tau = 0.0;
    for(int i = 0; i < x.rows(); i++){
        tau = 0.0;
        for(int d = 0; d < x.cols(); d++){
            tau += pow(xstar1d(d) - x(i,d), 2.0) / ls_sq(d);
        }
        c(i) = amp_sq * exp(-0.5 * tau);
    }
    return c;
}

// EQ kernel 1d cross covariance partial derivatives
//
// Inputs:
//     xstar1d : (d) array containing a single sample point
//     x : (n, d) array of sample points
//     d : dimension in which to compute the partial derivative
//     amp_sq : squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array of dimensionwise squared lengthscales
//     kxstarx : (n) array containing k(xstar1d, x | amp_sq, ls_sq)
//
// Outputs:
//     dK_1d_dx : (n) array containing the partial derivatives of
//         k(xstar1d, x | amp_sq, ls_sq) with respect to x in dimension d
//
Eigen::RowVectorXd cross_cov_grad_eq(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,
    Eigen::Ref<const Eigen::MatrixXd> x,
    int d,
    double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    Eigen::Ref<const Eigen::MatrixXd> kxstarx
    )
{
    return ((xstar1d(d) - x.col(d).array())
            / ls_sq(d)).transpose().array() * kxstarx.array();
}

// Class for a Gaussian Process with an EQ covariance kernel
//
// Private Attributes:
//     x : (n, d) array of sample points
//     y : (n) array of sample responses
//     amp_sq : squared marginal standard deviation (marginal variance)
//     ls_sq : (d) array of dimensionwise squared lengthscales
//     sigma_sq : squared Gaussian noise standard deviation (noise variance)
//     delta : stability jitter ("nugget")
//     lxxs : List of t memoized (n, n) sample covariance cholesky factors
//     axxs : List of t memoized (n) dot products between the inverse sample
//         covariance matrix and the sample responses.
//
//  Public Attributes:
//      xdata : raw (n,d) array of sample points
//      ydata : raw (n) array of sample responses
//      amp : (t) array of marginal standard deviation samples
//      ls : (t, d) array of dimensionwise (anisotropic) lengthscale samples
//      sigma : (t) array of Gaussian noise variance samples
//      jitter : stability jitter
//
//  Public Methods:
//      view_x : view the data x
//      view_y : view the data y
//      view_parameters : view the hyperparameters (amp_sq, ls_sq, sigma_sq)
//      view_lxx : view lower Cholesky factors of k(x,x)
//      posterior : posterior joint distribution
//      posterior_chol : posterior joint distribution (Cholesky parametrisation)
//      marginals : posterior marginals
//      E : posterior expectation
//      px : pure exploration (negative posterior variance)
//      px_jac : pure exploration and its jacobian
//      ei : expected improvement
//      ei_jac : expected improvement and its jacobian
//      ucb : upper confidence bound
//      ucb_jac : upper confidence bound and its jacobian
//      loo_cv : leave one out cross validation scores for sample data
//      entropy : differential entropy
//     
class GpEqKernel
{
private:
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
    Eigen::VectorXd amp_sq;
    Eigen::MatrixXd ls_sq;
    Eigen::VectorXd sigma_sq;
    double delta;
    std::vector<Eigen::LLT<Eigen::MatrixXd>> lxxs;
    std::vector<Eigen::VectorXd> axxs;

public:
    GpEqKernel(
        const Eigen::MatrixXd &xdata,
        const Eigen::VectorXd &ydata,
        const Eigen::VectorXd &amp,
        const Eigen::MatrixXd &ls,
        const Eigen::VectorXd &sigma,
        const double jitter
        );

    const Eigen::MatrixXd
    view_x();

    const Eigen::VectorXd
    view_y();

    std::tuple<const Eigen::VectorXd, const Eigen::MatrixXd, const Eigen::VectorXd>
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

// EQ kernel GP class-constructor
//
// Inputs:
//     xdata : raw (n,d) array of sample points
//     ydata : raw (n) array of sample responses
//     amp : (t) array of marginal standard deviation samples
//     ls : (t, d) array of dimensionwise (anisotropic) lengthscale samples
//     sigma : (t) array of Gaussian noise variance samples
//     jitter : stability jitter
//
// Outputs:
//     GpEqKernel : EQ kernel GP
//
// Notes:
//     Class constructor wants non-squared hyperparameters.
//
GpEqKernel::GpEqKernel(
    const Eigen::MatrixXd &xdata,
    const Eigen::VectorXd &ydata,
    const Eigen::VectorXd &amp,
    const Eigen::MatrixXd &ls,
    const Eigen::VectorXd &sigma,
    const double jitter
    ):
        x(xdata),
        y(ydata),
        amp_sq(amp.array().square()),
        ls_sq(ls.array().square()),
        sigma_sq(sigma.array().square()),
        delta(jitter)
{
    for (int k=0; k<amp_sq.size(); k++){
        lxxs.reserve(amp_sq.size());
        axxs.reserve(amp_sq.size());
        lxxs.push_back(
            L_cov_eq(
                x,
                amp_sq(k),
                ls_sq.row(k),
                sigma_sq(k) + delta)
            );
        axxs.push_back(lxxs[k].solve(y));
    }
}

// view the data x
//
// Outputs:
//     (n, d) array of sample points
//
const Eigen::MatrixXd
GpEqKernel::view_x()
{
    return x;
}

// view the daya y
//
// Outputs:
//     (n) array of sample responses
//
const Eigen::VectorXd
GpEqKernel::view_y()
{
    return y;
}

// view the hyperparameters
//
// Outputs:
//    amp_sq : (t) array of t squared marginal standard deviation samples
//    ls_sq : (t, d) array of t anisotropic squared lengthscale samples
//    sigma_sq : (t) array of t squared Gaussian noise standard deviation samples
//
std::tuple<const Eigen::VectorXd, const Eigen::MatrixXd, const Eigen::VectorXd>
GpEqKernel::view_parameters()
{
    return std::make_tuple(amp_sq, ls_sq, sigma_sq);
}

// view the lower Cholesky factors of k(x, x | amp, ls, sigma)
//
// Outputs:
//     lxxs : length t list of (n,n) arrays containing the lower Cholesky factor
//         of K = k(x, x | amp_t, ls_t, sigma_t)
//
std::vector<Eigen::MatrixXd> GpEqKernel::view_lxx()
{
    std::vector<Eigen::MatrixXd> lxx_views;
    lxx_views.reserve(amp_sq.size());
    for (int k=0; k<amp_sq.size(); k++){
        lxx_views.push_back(lxxs[k].matrixL());
    }
    return lxx_views;
}

// posterior joint distribution at xstar
//
// Inputs:
//     xstar : (m, d) array of points at which to compute the posterior
//         (equivalently, "conditional" or "predictive") distribution
//
// Outputs:
//     mu : (m, t) array containing t length m vectors corresponding to the
//         posterior mean at xstar for each of the t hyperparameter samples.
//     K : Length t list of (m, m) arrays corresponding to the posterior
//         covariance matrices for each of the t hyperparameter samples.
//
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>>
GpEqKernel::posterior(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::MatrixXd m(xstar.rows(), amp_sq.size());
    std::vector<Eigen::MatrixXd> v;
    v.reserve(amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(
            x,
            xstar,
            amp_sq(k),
            ls_sq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        // next we compute:
        //
        // kzz - kzx * kxx_inv * kxz
        //
        // but reference only the lower triangular half and return
        // a symmetrical view
        v.push_back(
            (
                 cov_eq(
                     xstar,
                     amp_sq(k),
                     ls_sq.row(k),
                     sigma_sq(k)).array() -
                 (
                     (kxxstar.transpose() * lxxs[k].solve(kxxstar))
                     ).array()
                 ).matrix().selfadjointView<Eigen::Lower>()
            );
    }
    return std::make_tuple(m, v);
}


// posterior distribution at xstar (Cholesky parametrisation)
//
// Inputs:
//     xstar : (m, d) array of points at which to compute the posterior
//         (equivalently, "conditional" or "predictive") distribution
//
// Outputs:
//     mu : (m, t) array containing t length m vectors corresponding to the
//         posterior mean at xstar for each of the t hyperparameter samples.
//     L : Length t list of (m, m) arrays corresponding to the lower Cholesky
//         factors posterior of the covariance matrices for each of the t
//          hyperparameter samples.
//
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>>
GpEqKernel::posterior_chol(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::MatrixXd m(xstar.rows(), amp_sq.size());
    std::vector<Eigen::MatrixXd> v;
    v.reserve(amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        // next we compute:
        //
        // kzz - kzx * kxx_inv * kxz
        //
        // but reference only the lower triangular half and return
        // a symmetrical view
        v.push_back(
            (
                cov_eq(
                    xstar,
                    amp_sq(k),
                    ls_sq.row(k),
                    sigma_sq(k)).array() -
                (
                    (kxxstar.transpose() * lxxs[k].solve(kxxstar))
                    ).array()
                ).matrix().selfadjointView<Eigen::Lower>().llt().matrixL()
            );
    }
    return std::make_tuple(m, v);
}

// posterior marginals at xstar
//
// Inputs:
//     xstar : (m, d) array of points at which to compute the posterior
//         (equivalently, "conditional" or "predictive") marginals
//
// Outputs:
//     mu : (m, t) array containing t length m vectors corresponding to the
//         posterior mean at xstar for each of the t hyperparameter samples.
//     var : (m, t) array containing t length m vectors corresponding to the
//         posterior marginal variance at xstar for each of the t hyperparameter
//         samples.
//        
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>
GpEqKernel::marginals(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::MatrixXd m(xstar.rows(), amp_sq.size());
    Eigen::MatrixXd v(xstar.rows(), amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        // next, compute diag(kzz) - diag(kzx * kxx_inv * kxz)
        // where:
        //     diag(kzz) = amp_sq + sigma_sq
        // and:
        //     kxx_inv = lxx \ kxz
        //
        // then ensure all elements > delta
        v.col(k) = (
            (amp_sq(k) + sigma_sq(k)) -
            ((kxxstar.transpose() * lxxs[k].solve(kxxstar))
             .diagonal()).array()
            ).cwiseMax(delta).matrix();
    }
    return std::make_tuple(m, v);
}

// posterior expectation
//
// Inputs:
//     xstar : (m, d) array of points at which to compute the posterior
//         expectation.
//
// Outputs:
//     E : (m) array containing the posterior expectation at xstar, integrated
//         over the hyperparameter samples.
//
Eigen::VectorXd
GpEqKernel::E(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::VectorXd m = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(
            x,
            xstar,
            amp_sq(k),
            ls_sq.row(k));
        m.noalias() += (kxxstar.transpose() * axxs[k]);
    }
    return m / amp_sq.size();
}

// posterior expectation and jacobian
//
// Inputs:
//     xstar1d : (d) array representing a single d-dimensional point at which
//         the posterior expectation and its jacobian should be computed.
//        
// Outputs:
//     E : float describing the posterior expectation at xstar1d, integrated
//         over the hyperparameter samples.
//     E_jac : (d) array containing the partial derivatives of E at xstar1d with
//         respect to xstar1d
//
std::tuple<double, Eigen::VectorXd>
GpEqKernel::E_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar
    )
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    double m = 0;
    Eigen::VectorXd jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_eq(
            xstar,
            x,
            amp_sq(k),
            ls_sq.row(k));
        m += kxstarx.dot(axxs[k]);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_eq(
                xstar,
                x,
                d,
                amp_sq(k),
                ls_sq.row(k),
                kxstarx);
            jac(d) -= dkxstarx_dxstar.dot(axxs[k]);
        }
    }
    return std::make_tuple(m / amp_sq.size(), jac / amp_sq.size());
}

// pure exploration acquisition function
//
// Inputs:
//     xstar : (m, d) array of points at which to compute the posterior
//         expectation.
//
// Outputs:
//     px : (m) array containing the pure exploration at xstar
//
// Notes:
//     px is equivalent to the (negative) expectation of the marginal variance
//
Eigen::VectorXd
GpEqKernel::px(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd v = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(
            x,
            xstar,
            amp_sq(k),
            ls_sq.row(k));
        v.noalias() -=
            ((amp_sq(k) + sigma_sq(k)) -
             ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()
                 ).array()).cwiseMax(delta).matrix();
    }
    return v / amp_sq.size();
}

// pure exploration and jacobian
//
// Inputs:
//     xstar1d : (d) array representing a single d-dimensional point at which to
//         the pure exploration acquisition function and its jacobian should be
//         computed.
//
// Outputs:
//     px : float describing the pure exploration acquisition function at
//         xstar1d, integrated over the hyperparameter samples.
//     px_jac : (d) array containing the partial derivatives of the pure
//         exploration at xstar1d with respect to xstar1d.
//
// Notes:
//     px is equivalent to the (negative) expectation of the marginal variance
//
std::tuple<Eigen::VectorXd, Eigen::VectorXd>
GpEqKernel::px_jac(
    Eigen::Ref<const Eigen::RowVectorXd> xstar
    )
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x.rows(), x.rows());
    Eigen::RowVectorXd v = Eigen::VectorXd::Zero(1);
    Eigen::MatrixXd jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_eq(
            xstar,
            x,
            amp_sq(k),
            ls_sq.row(k));
        v.noalias() -=
            ((amp_sq(k) + sigma_sq(k)) -
             ((kxstarx * lxxs[k].solve(kxstarx.transpose())).diagonal()
                 ).array()).cwiseMax(delta).matrix();
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_eq(
                xstar,
                x,
                d,
                amp_sq(k),
                ls_sq.row(k),
                kxstarx);
            jac(d) += (kxstarx * lxxs[k].solve(-dkxstarx_dxstar.transpose()));
        }
    }
    return std::make_tuple(v / amp_sq.size(), jac*2 / amp_sq.size());
}

// expected improvement
//
// Inputs:
//     xstar : (m, d) array of points at which to compute the expected
//         improvement
//
// Outputs:
//     ei : (m) array containing the expected improvement at xstar,
//         integrated over the hyperparameter samples
//
// Notes:
//     This implementation of ei is negative and seeks the minimum
//
Eigen::VectorXd
GpEqKernel::ei(
    Eigen::Ref<const Eigen::MatrixXd> xstar
    )
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ei = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::VectorXd gamma(xstar.rows());
    double ystar = y.minCoeff();
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(
            x,
            xstar,
            amp_sq(k),
            ls_sq.row(k));
        mean = kxxstar.transpose() * axxs[k];
        sd = (
            (amp_sq(k) + sigma_sq(k)) -
            ((kxxstar.transpose() * lxxs[k].solve(kxxstar)
                ).diagonal()).array()).cwiseMax(delta).sqrt().matrix();
        gamma = (ystar - mean.array()).array() / sd.array();
        ei.array() -= (
            (ystar - mean.array()) * ncdf(gamma).array() +
            sd.array() * npdf(gamma).array()
            ).array();
    }
    return ei / amp_sq.size();
}

// expected improvement and jacobian
//
// Inputs:
//     xstar1d : (d) array representing a single d-dimensional point at which to
//         the expected improvement and its jacobian should be computed.
//
// Outputs:
//     ei : float describing the expected improvement at xstar1d, integrated
//         over the hyperparameter samples.
//     ei_jac : (d) array containing the partial derivatives of the expected
//         improvement at xstar1d with respect to xstar1d.
//
// Notes:
//     this implementation of ei is negative and seeks the minimum
//
std::tuple<double, Eigen::VectorXd>
GpEqKernel::ei_jac(
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
    Eigen::VectorXd jac(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_eq(
            xstar,
            x,
            amp_sq(k),
            ls_sq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (amp_sq(k) + sigma_sq(k)) -
            kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        gamma = (ystar - mean) / sd;
        ei -= (ystar - mean) * ncdfd(gamma) + sd * npdfd(gamma);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_eq(
                xstar,
                x,
                d,
                amp_sq(k),
                ls_sq.row(k),
                kxstarx);
            sd_grad = -(
                kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose()))
                ) / sd;
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            gamma_grad = (
                ((mean - ystar) * sd_grad) - (sd * mean_grad)
                ) / var;
            jac(d) -= (ystar-mean) *
                gamma_grad * npdfd(gamma) -
                ncdfd(gamma) * mean_grad +
                npdfd(gamma) * sd_grad + sd * gamma_grad * npdfdg(gamma);
        }
    }
    return std::make_tuple(ei / amp_sq.size(), jac / amp_sq.size());
}

// upper confidence bound
//
// Inputs:
//     xstar : (m, d) array of points at which to compute the expected
//         improvement
//     beta : ucb beta parameter
//
// Outputs:
//     ucb : (m) array containing the upper confidence bound at xstar,
//         integrated over the hyperparameter samples
//
// Notes:
//     This implementation of ucb is negative and seeks the minimum
//
Eigen::VectorXd
GpEqKernel::ucb(
    Eigen::Ref<const Eigen::MatrixXd> xstar,
    double beta
    )
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ucb = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(
            x,
            xstar,
            amp_sq(k),
            ls_sq.row(k));
        mean = kxxstar.transpose() * axxs[k];
        sd = ((amp_sq(k) + sigma_sq(k)) -
              ((kxxstar.transpose() * lxxs[k].solve(kxxstar)
                  ).diagonal()).array()).cwiseMax(delta).sqrt().matrix();
        ucb.array() += mean.array() - (sqrt(beta) * sd.array());
    }
    return ucb / amp_sq.size();
}

// upper confidence bound and jacobian
//
// Inputs:
//     xstar1d : (d) array representing a single d-dimensional point at which to
//         the upper confidence bound and its jacobian should be computed.
//     beta : ucb beta parameter
//
// Outputs:
//     ucb : float describing the expected improvement at xstar1d, integrated
//         over the hyperparameter samples.
//     ucb_jac : (d) array containing the partial derivatives of ucb at xstar1d
//         with respect to xstar1d.
//
// Notes:
//     this implementation of ucb is negative and seeks the minimum
//
std::tuple<double, Eigen::VectorXd>
GpEqKernel::ucb_jac(
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
    double srbeta = sqrt(beta);
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    Eigen::VectorXd jac(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_eq(
            xstar,
            x,
            amp_sq(k),
            ls_sq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (amp_sq(k) + sigma_sq(k)) -
            kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        ucb += mean - (srbeta * sd);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_eq(
                xstar,
                x,
                d,
                amp_sq(k),
                ls_sq.row(k),
                kxstarx);
            sd_grad = -(
                kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose()))
                ) / sd;
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            jac(d) += mean_grad - (srbeta * sd_grad);
        }
    }
    return std::make_tuple(ucb / amp_sq.size(), jac / amp_sq.size());
}

// leave-one-out predictive sample data density
//
// Outputs:
//     loo_cv : (n, t) array containing the loo-cv scores for each of the n
//     training samples and each of the t hyperparameter samples
//
// Notes:
//     Uses the method of Sundararajan and Keerthi:
//     Predictive approaches for choosing hyperparameters in Gaussian Processes.
//     Advances in Neural Information Processing Systems 631-637
//     2000
//
Eigen::MatrixXd GpEqKernel::loo_cv()
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
//     xstar : (m, d) array of points at which to compute the differential
//         entropy
//
// Outputs:
//     h : The differential entropy of the GP posterior calculated at the
//         points containing in xstar, integrated over the t hyperparameter
//         samples.
double GpEqKernel::entropy(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    const double e = std::exp(1.0);
    double h = 0;
    double tmp = xstar.rows() / 2.0 *  log(e * M_PI * 2.0);
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::MatrixXd c(xstar.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(
            x,
            xstar,
            amp_sq(k),
            ls_sq.row(k));
        //computes
        //
        // h = tmp + 0.5 * 2 * log(det(Sigma))
        // = tmp + 0.5 * 2 * log(det(kzz - kzx * kxx_inv * kxz))
        // = tmp + sum(log(diag(chol(kzz - kzx * kxx_inv * kxz))))
        //
        //in a way that Eigen can (hopefully) optimise
        h += tmp + (
            cov_eq(
                xstar,
                amp_sq(k),
                ls_sq.row(k),
                sigma_sq(k)
                ).array() -
            ((kxxstar.transpose() * lxxs[k].solve(kxxstar))).array()
            ).matrix().llt().matrixLLT().diagonal().array().log().sum();
    }
    return h / amp_sq.size();
}
