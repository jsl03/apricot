/* Exponentiated Quadratic (EQ) Kernel*/

Eigen::LLT<Eigen::MatrixXd> L_cov_eq(
    Eigen::Ref<const Eigen::MatrixXd> x,      // (n,d) array of points 
    const double amp_sq,                      // squared marginal standard deviation 
    Eigen::Ref<const Eigen::VectorXd> ls_sq,  // (d,) squared anisotropic lengthscales
    const double sigma_sq                     // squared noise standard deviation
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
    return c.selfadjointView<Eigen::Lower>().llt();
}

Eigen::MatrixXd cov_eq(
    Eigen::Ref<const Eigen::MatrixXd> x,      // (n,d) array of points 
    const double amp_sq,                      // squared marginal standard deviation
    Eigen::Ref<const Eigen::VectorXd> ls_sq,  // (d,) squared anisotropic lengthscales
    const double sigma_sq                     // squared noise standard deviation
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

Eigen::MatrixXd cross_cov_eq(
    Eigen::Ref<const Eigen::MatrixXd> x1,     // (n,d) array of points
    Eigen::Ref<const Eigen::MatrixXd> x2,     // (n,d) array of points
    double amp_sq,                            // squared marginal standard deviation
    Eigen::Ref<const Eigen::VectorXd> ls_sq   // (d,) squared anisotropic lengthscales
)
{
    Eigen::RowVectorXd ls = ls_sq.transpose().array().sqrt();
    Eigen::MatrixXd tau(x1.rows(), x2.rows());
    Eigen::MatrixXd al = x1.array().rowwise() / ls.array();
    Eigen::VectorXd tmp1 = al.array().square().rowwise().sum();
    Eigen::MatrixXd bl = x2.array().rowwise() / ls.array();
    Eigen::VectorXd tmp2 = bl.array().square().rowwise().sum();
    tau = tmp1.rowwise().replicate(tmp2.size()) + tmp2.transpose().colwise().replicate(tmp1.size());
    tau.noalias() += -2.0 * al * bl.transpose();
    return amp_sq * ((-0.5 * tau).array().exp()).matrix();
}

Eigen::RowVectorXd cov_1d_eq(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,    // (d,) vector 
    Eigen::Ref<const Eigen::MatrixXd> x,             // (n, d) array of points
    const double amp_sq,                             // squared marginal standard deviation
    Eigen::Ref<const Eigen::VectorXd> ls_sq)         // (d,) squared anisotropic lengthscales
{
    Eigen::RowVectorXd c(x.rows());
    double tau;
    for(int i = 0; i < x.rows(); i++){
        tau = 0;
        for(int d = 0; d < x.cols(); d++){
            tau += pow(xstar1d(d) - x(i,d), 2.0) / ls_sq(d);
        }
        c(i) = amp_sq * exp(-0.5 * tau);
    }
    return c;
}

Eigen::RowVectorXd cross_cov_grad_eq(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,   // (d,) vector
    Eigen::Ref<const Eigen::MatrixXd> x,            // (n, d) array of points
    int d,                                          // dimension to compute derivative in
    double amp_sq,                                  // squared marginal standard deviation
    Eigen::Ref<const Eigen::VectorXd> ls_sq,        // (d) squared anisotropic lengthscales
    Eigen::Ref<const Eigen::MatrixXd> kxstarx)      // cross_cov_eq(xstar, x | amp_sq, ls_sq)
{
    return ((xstar1d(d) - x.col(d).array()) / ls_sq(d)).transpose().array() * kxstarx.array();
}

/* Gaussian Process Class*/

class GpEqKernel
{
private:
    Eigen::MatrixXd x;                                  // (n, d) array of points
    Eigen::VectorXd y;                                  // (n) vector of responses
    Eigen::VectorXd amp_sq;                             // marginal standard deviation
    Eigen::MatrixXd ls_sq;                              // (d) squared anisotropic lengthscales
    Eigen::VectorXd sigma_sq;                           // marginal noise variance
    double delta;                                       // stability jitter
    std::vector<Eigen::LLT<Eigen::MatrixXd>> lxxs;      // lower cholesky factor of covariance matrix
    std::vector<Eigen::VectorXd> axxs;                  // dot product between inverse covariance matrix and vector y

public:
    GpEqKernel(
        const Eigen::MatrixXd &xdata,  // (n, d) array of points
        const Eigen::VectorXd &ydata,  // (n) vector of responses
        const Eigen::VectorXd &amp,    // (m) samples of marginal standard deviation
        const Eigen::MatrixXd &ls,     // (m, d) array of m samples of the d anisotropic lengthscales
        const Eigen::VectorXd &sigma,  // (m) samples of the marginal noise standard deviation
        const double jitter            // stability jitter
        );

    // view sample points (i.e., array x)
    const Eigen::MatrixXd view_x();

    // view responses (i.e., vector y)
    const Eigen::VectorXd view_y();

    // view hyperparameters (i.e., amp, ls, sigma)
    std::tuple<const Eigen::VectorXd, const Eigen::MatrixXd, const Eigen::VectorXd> view_parameters();

    // view lower cholesky factors of k(x, x | amp, ls, sigma)
    std::vector<Eigen::MatrixXd> view_lxx();

    // posterior joint distribution
    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> posterior(Eigen::Ref<const Eigen::MatrixXd> xstar);

    // lower cholesky factor of posterior joint distribution
    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> posterior_chol(Eigen::Ref<const Eigen::MatrixXd> xstar);

    // posterior marginal distributions
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> marginals(Eigen::Ref<const Eigen::MatrixXd> xstar);

    // expectation
    Eigen::VectorXd E(Eigen::Ref<const Eigen::MatrixXd> xstar);

    // expectation and its jacobian
    std::tuple<double, Eigen::VectorXd> E_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar);

    // pure exploration (i.e., posterior variance)
    Eigen::VectorXd px(Eigen::Ref<const Eigen::MatrixXd> xstar);

    // pure exploration and its jacobian
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> px_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar);

    // expected improvement
    Eigen::VectorXd ei(Eigen::Ref<const Eigen::MatrixXd> xstar);

    // expected improvement and its jacobian
    std::tuple<double, Eigen::VectorXd> ei_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar);

    // upper confidence bound
    Eigen::VectorXd ucb(Eigen::Ref<const Eigen::MatrixXd> xstar, double beta);

    // upper confidence bound and its jacobian
    std::tuple<double, Eigen::VectorXd> ucb_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar, double beta);

    // leave-one-out cross validation
    Eigen::MatrixXd loo_cv();

    // differential entropy
    double entropy(Eigen::Ref<const Eigen::MatrixXd> xstar);
};

// class-constructor
GpEqKernel::GpEqKernel(
    const Eigen::MatrixXd &xdata,
    const Eigen::VectorXd &ydata,
    const Eigen::VectorXd &amp,
    const Eigen::MatrixXd &ls,
    const Eigen::VectorXd &sigma,
    const double jitter):
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
        lxxs.push_back(L_cov_eq(x, amp_sq(k), ls_sq.row(k), sigma_sq(k) + delta));
        axxs.push_back(lxxs[k].solve(y));
    }
}


// provide a view of x data
const Eigen::MatrixXd GpEqKernel::view_x()
{
    return x;
}

// provide a view of y data
const Eigen::VectorXd GpEqKernel::view_y()
{
    return y;
}

// provide a view of the hyperparameters (amplitude, lengthscales, noise s.d.)
std::tuple<const Eigen::VectorXd, const Eigen::MatrixXd, const Eigen::VectorXd> GpEqKernel::view_parameters()
{
    return std::make_tuple(amp_sq, ls_sq, sigma_sq);
}

// provide a view of covariance matrix cholesky factors
std::vector<Eigen::MatrixXd> GpEqKernel::view_lxx()
{
    std::vector<Eigen::MatrixXd> lxx_views;
    lxx_views.reserve(amp_sq.size());
    for (int k=0; k<amp_sq.size(); k++){
        lxx_views.push_back(lxxs[k].matrixL());
    }
    return lxx_views;
}

// Posterior
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> GpEqKernel::posterior(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd m(xstar.rows(), amp_sq.size());
    std::vector<Eigen::MatrixXd> v;
    v.reserve(amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        /*
        computes:
        kzz - kzx * kxx_inv * kxz

        but references only the lower triangular half and returns
        a symmetrical view
        */
        v.push_back(
             (
                 cov_eq(xstar, amp_sq(k), ls_sq.row(k), sigma_sq(k)).array() -
                 ((kxxstar.transpose() * lxxs[k].solve(kxxstar))).array()
             )
             .matrix().selfadjointView<Eigen::Lower>()
        );
    }
    return std::make_tuple(m, v);
}

// Posterior (Cholesky)
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> GpEqKernel::posterior_chol(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd m(xstar.rows(), amp_sq.size());
    std::vector<Eigen::MatrixXd> v;
    v.reserve(amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        /*
          computes:
          kzz - kzx * kxx_inv * kxz

          but references only the lower triangular half and returns
          a symmetrical view
        */
        v.push_back(
            (
                cov_eq(xstar, amp_sq(k), ls_sq.row(k), sigma_sq(k)).array() -
                ((kxxstar.transpose() * lxxs[k].solve(kxxstar))).array()
            )
            .matrix().selfadjointView<Eigen::Lower>().llt().matrixL()
        );
    }
    return std::make_tuple(m, v);
}

// Posterior marginals
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> GpEqKernel::marginals(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd m(xstar.rows(), amp_sq.size());
    Eigen::MatrixXd v(xstar.rows(), amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        // also ensures posterior variance > delta
        v.col(k) = ((amp_sq(k) + sigma_sq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).matrix();
    }
    return std::make_tuple(m, v);
}

// Expectation and jacobian
Eigen::VectorXd GpEqKernel::E(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::VectorXd m = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        m.noalias() += (kxxstar.transpose() * axxs[k]);
    }
    return m / amp_sq.size();
}

std::tuple<double, Eigen::VectorXd> GpEqKernel::E_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar)
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    double m = 0;
    Eigen::VectorXd jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_eq(xstar, x, amp_sq(k), ls_sq.row(k));
        m += kxstarx.dot(axxs[k]);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_eq(xstar, x, d, amp_sq(k), ls_sq.row(k), kxstarx);
            jac(d) -= dkxstarx_dxstar.dot(axxs[k]);
        }
    }
    return std::make_tuple(m / amp_sq.size(), jac / amp_sq.size());
}

// Pure Exploration objective and jacobian
Eigen::VectorXd GpEqKernel::px(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd v = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        v.noalias() -= ((amp_sq(k) + sigma_sq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).matrix();
    }
    return v / amp_sq.size();
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> GpEqKernel::px_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar)
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x.rows(), x.rows());
    Eigen::RowVectorXd v = Eigen::VectorXd::Zero(1);
    Eigen::MatrixXd jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_eq(xstar, x, amp_sq(k), ls_sq.row(k));
        v.noalias() -= ((amp_sq(k) + sigma_sq(k)) - ((kxstarx * lxxs[k].solve(kxstarx.transpose())).diagonal()).array()).cwiseMax(delta).matrix();
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_eq(xstar, x, d, amp_sq(k), ls_sq.row(k), kxstarx);
            jac(d) += (kxstarx * lxxs[k].solve(-dkxstarx_dxstar.transpose()));
        }
    }
    return std::make_tuple(v / amp_sq.size(), jac*2 / amp_sq.size());
}

// Expected improvement objective and jacobian
Eigen::VectorXd GpEqKernel::ei(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ei = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::VectorXd gamma(xstar.rows());
    double ystar = y.minCoeff();
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        mean = kxxstar.transpose() * axxs[k];
        sd = ((amp_sq(k) + sigma_sq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).sqrt().matrix();
        gamma = (ystar - mean.array()).array() / sd.array();
        ei.array() -= ((ystar - mean.array()) * ncdf(gamma).array() + sd.array() * npdf(gamma).array()).array();  
    }
    return ei / amp_sq.size();
}

std::tuple<double, Eigen::VectorXd> GpEqKernel::ei_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar)
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
        kxstarx = cov_1d_eq(xstar, x, amp_sq(k), ls_sq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (amp_sq(k) + sigma_sq(k)) - kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        gamma = (ystar - mean) / sd;
        ei -= (ystar - mean) * ncdfd(gamma) + sd * npdfd(gamma);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_eq(xstar, x, d, amp_sq(k), ls_sq.row(k), kxstarx);
            sd_grad = -(kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose()))) / (sd);
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            gamma_grad = (((mean - ystar) * sd_grad) - (sd * mean_grad)) / var;
            jac(d) -= (ystar-mean) * gamma_grad * npdfd(gamma) - ncdfd(gamma) * mean_grad + npdfd(gamma)*sd_grad + sd * gamma_grad * npdfdg(gamma);
        }
    }
    return std::make_tuple(ei/amp_sq.size(), jac/amp_sq.size());
}

// upper confidence bound objective and jacobian
Eigen::VectorXd GpEqKernel::ucb(Eigen::Ref<const Eigen::MatrixXd> xstar, double beta)
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ucb = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        mean = kxxstar.transpose() * axxs[k];
        sd = ((amp_sq(k) + sigma_sq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).sqrt().matrix();
        ucb.array() += mean.array() - (sqrt(beta) * sd.array());
    }
    return ucb / amp_sq.size();
}

std::tuple<double, Eigen::VectorXd> GpEqKernel::ucb_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar, double beta)
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
        kxstarx = cov_1d_eq(xstar, x, amp_sq(k), ls_sq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (amp_sq(k) + sigma_sq(k)) - kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        ucb += mean - (srbeta * sd);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_eq(xstar, x, d, amp_sq(k), ls_sq.row(k), kxstarx);
            sd_grad = -(kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose()))) / (sd);
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            jac(d) += mean_grad - (srbeta * sd_grad);
        }
    }
    return std::make_tuple(ucb/amp_sq.size(), jac/amp_sq.size());
}

// LOO_CV predictive density
Eigen::MatrixXd GpEqKernel::loo_cv()
{
    Eigen::MatrixXd cv(amp_sq.rows(), x.rows());
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x.rows(), x.rows());
    Eigen::VectorXd c_diag;
    for (int k=0; k<amp_sq.size(); k++){
        c_diag = lxxs[k].solve(I).diagonal();
        cv.row(k) = -0.5*log(2*M_PI) + (0.5*(c_diag.array().log())) - 0.5 * (axxs[k].array().square() / c_diag.array());
    }
    return cv;
}

// Entropy
double GpEqKernel::entropy(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    const double e = std::exp(1.0);
    double h = 0;
    double tmp = xstar.rows() / 2.0 *  log(e * M_PI * 2.0);
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::MatrixXd c(xstar.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_eq(x, xstar, amp_sq(k), ls_sq.row(k));
        /*
        computes

        h = tmp + 0.5 * 2 * log(det(Sigma))
          = tmp + 0.5 * 2 * log(det(kzz - kzx * kxx_inv * kxz))
          = tmp + sum(log(diag(chol(kzz - kzx * kxx_inv * kxz))))
  
        in a way that (hopefully) lets Eigen optimise the calculation
        */
        h += tmp + ( cov_eq(xstar, amp_sq(k), ls_sq.row(k), sigma_sq(k)).array() -
                   ((kxxstar.transpose() * lxxs[k].solve(kxxstar))
                   ).array()).matrix().llt().matrixLLT().diagonal().array().log().sum();

    }
    return h / amp_sq.size();
}
