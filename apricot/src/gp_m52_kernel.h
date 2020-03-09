/* Matern52 (M52) Covariance Kernel */

Eigen::LLT<Eigen::MatrixXd> L_cov_m52(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    const double sigma_sq)
{
    Eigen::MatrixXd c(x.rows(), x.rows());
    double tau;
    double sr_tau;
    double sr_5 = sqrt(5.0);
    double rat = 5.0 / 3.0;
    for(int i = 0; i < x.rows(); i++){
        for(int j = i; j < x.rows(); j++){
            if (i!=j){
                tau = 0.0;
                for(int d = 0; d < x.cols(); d++){
                    tau += pow(x(j, d) - x(i, d), 2.0) / ls_sq(d);
                }
                sr_tau = sqrt(tau);
                c(j, i) = amp_sq * (1 + sr_5 * sr_tau + rat * tau) * exp(-sr_5 * sr_tau);
            }
            if (i==j){
                c(i,i) = amp_sq + sigma_sq;
            }
        }
    }
    return c.selfadjointView<Eigen::Lower>().llt();
}

Eigen::MatrixXd cov_m52(
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq,
    const double sigma_sq)
{
    Eigen::MatrixXd c(x.rows(), x.rows());
    double tau;
    double sr_tau;
    double sr_5 = sqrt(5.0);
    double rat = 5.0 / 3.0;
    for(int i = 0; i < x.rows(); i++){
        for(int j = i; j < x.rows(); j++){
            if (i!=j){
                tau = 0.0;
                for(int d = 0; d < x.cols(); d++){
                    tau += pow(x(j, d) - x(i, d), 2.0) / ls_sq(d);
                }
                sr_tau = sqrt(tau);
                c(j, i) = amp_sq * (1 + sr_5 * sr_tau + rat * tau) * exp(-sr_5 * sr_tau);
            }
            if (i==j){
                c(i,i) = amp_sq + sigma_sq;
            }
        }
    }
    return c.selfadjointView<Eigen::Lower>();
}

Eigen::MatrixXd cross_cov_m52(
    Eigen::Ref<const Eigen::MatrixXd> x1,
    Eigen::Ref<const Eigen::MatrixXd> x2,
    double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq)
{
    double sr_5 = sqrt(5);
    double rat = 5.0 / 3.0;
    Eigen::RowVectorXd ls = ls_sq.transpose().array().sqrt();
    Eigen::MatrixXd tau(x1.rows(), x2.rows());
    Eigen::MatrixXd sr_tau(x1.rows(), x2.rows());
    Eigen::MatrixXd al = x1.array().rowwise() / ls.array();
    Eigen::VectorXd tmp1 = al.array().square().rowwise().sum();
    Eigen::MatrixXd bl = x2.array().rowwise() / ls.array();
    Eigen::VectorXd tmp2 = bl.array().square().rowwise().sum();
    tau = tmp1.rowwise().replicate(tmp2.size()) + tmp2.transpose().colwise().replicate(tmp1.size());
    tau.noalias() += -2.0 * al * bl.transpose();
    sr_tau = tau.array().sqrt();
    return amp_sq * (1.0 + sr_5 * sr_tau.array() + rat * tau.array()) * (-sr_5 * sr_tau).array().exp();
}

Eigen::RowVectorXd cov_1d_m52(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,
    Eigen::Ref<const Eigen::MatrixXd> x,
    const double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq)
{
    Eigen::RowVectorXd c(x.rows());
    double tau;
    double sr_tau;
    double sr_5 = sqrt(5.0);
    double rat = 5.0 / 3.0;
    for(int i = 0; i < x.rows(); i++){
        tau = 0;
        for(int d = 0; d < x.cols(); d++){
            tau += pow(xstar1d(d) - x(i,d), 2.0) / ls_sq(d);
        }
        sr_tau = sqrt(tau);
        c(i) = amp_sq * (1.0 + sr_5 * sr_tau + rat * tau) * exp(-sr_5 * sr_tau);
    }
    return c;
}

Eigen::RowVectorXd cross_cov_grad_m52(
    Eigen::Ref<const Eigen::RowVectorXd> xstar1d,
    Eigen::Ref<const Eigen::MatrixXd> x,
    int d,
    double amp_sq,
    Eigen::Ref<const Eigen::VectorXd> ls_sq)
{
    Eigen::RowVectorXd dc(x.rows());
    double tau;
    double sr_5 = sqrt(5.0);
    double tmp_1;
    double tmp_2;
    double rat_10_3 = 10.0 /3.0;
    double rat_5_3 = 5.0 / 3.0;

    for(int i = 0; i < x.rows(); i++){
        tau = 0;
        for(int d_internal = 0; d_internal < x.cols(); d_internal++){
            tau += pow(xstar1d(d_internal) - x(i,d_internal), 2.0) / ls_sq(d_internal);
        }
        tmp_1 = sqrt(tau) * sr_5;
        tmp_2 = amp_sq * (rat_10_3 - 5.0 - rat_5_3 * tmp_1) * exp(-tmp_1);
        dc(i) = -((xstar1d(d) - x(i,d)) / ls_sq(d)) * tmp_2;
    }
    return dc;
}

/* Gaussian Process Class */

class GpM52Kernel
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
    GpM52Kernel(
        const Eigen::MatrixXd &xdata,
        const Eigen::VectorXd &ydata,
        const Eigen::VectorXd &amp,
        const Eigen::MatrixXd &ls,
        const Eigen::VectorXd &sigma,
        const double jitter);

    // data views
    const Eigen::MatrixXd view_x();
    const Eigen::VectorXd view_y();
    std::tuple<const Eigen::VectorXd, const Eigen::MatrixXd, const Eigen::VectorXd> view_parameters();
    std::vector<Eigen::MatrixXd> view_lxx();

    // posterior
    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> posterior(Eigen::Ref<const Eigen::MatrixXd> xstar);

    // posterior (cholesky decomposition)
    std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> posterior_chol(Eigen::Ref<const Eigen::MatrixXd> xstar);

    // posterior marginals
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> marginals(Eigen::Ref<const Eigen::MatrixXd> xstar);

    // expectation
    Eigen::VectorXd E(Eigen::Ref<const Eigen::MatrixXd> xstar);
    std::tuple<double, Eigen::VectorXd> E_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar);

    // pure exploration (i.e., posterior variance)
    Eigen::VectorXd px(Eigen::Ref<const Eigen::MatrixXd> xstar);
    std::tuple<Eigen::VectorXd, Eigen::VectorXd> px_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar);

    // expected improvement
    Eigen::VectorXd ei(Eigen::Ref<const Eigen::MatrixXd> xstar);
    std::tuple<double, Eigen::VectorXd> ei_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar);

    // upper confidence bound
    Eigen::VectorXd ucb(Eigen::Ref<const Eigen::MatrixXd> xstar, double beta);
    std::tuple<double, Eigen::VectorXd> ucb_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar, double beta);

    // leave-one-out cross validation
    Eigen::MatrixXd loo_cv();

    // entropy
    double entropy(Eigen::Ref<const Eigen::MatrixXd> xstar);
};

// class-constructor
GpM52Kernel::GpM52Kernel(
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
        lxxs.push_back(L_cov_m52(x, amp_sq(k), ls_sq.row(k), sigma_sq(k)));
        axxs.push_back(lxxs[k].solve(y));
    }
}


// provide a view of x data
const Eigen::MatrixXd GpM52Kernel::view_x()
{
    return x;
}

// provide a view of y data
const Eigen::VectorXd GpM52Kernel::view_y()
{
    return y;
}

// provide a view of parameters
std::tuple<const Eigen::VectorXd, const Eigen::MatrixXd, const Eigen::VectorXd> GpM52Kernel::view_parameters()
{
    return std::make_tuple(amp_sq, ls_sq, sigma_sq);
}

// provide a view of covariance matrix cholesky factors
std::vector<Eigen::MatrixXd> GpM52Kernel::view_lxx()
{
    std::vector<Eigen::MatrixXd> lxx_views;
    lxx_views.reserve(amp_sq.size());
    for (int k=0; k<amp_sq.size(); k++){
        lxx_views.push_back(lxxs[k].matrixL());
    }
    return lxx_views;
}

// Posterior
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> GpM52Kernel::posterior(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd m(xstar.rows(), amp_sq.size());
    std::vector<Eigen::MatrixXd> v;
    v.reserve(amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_m52(x, xstar, amp_sq(k), ls_sq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        /*
        computes:
        kzz - kzx * kxx_inv * kxz

        but references only the lower triangular half and returns
        a symmetrical view
        */
        v.push_back(
             (
                 cov_m52(xstar, amp_sq(k), ls_sq.row(k), sigma_sq(k)).array() -
                 ((kxxstar.transpose() * lxxs[k].solve(kxxstar))).array()
             )
             .matrix().selfadjointView<Eigen::Lower>()
        );
    }
    return std::make_tuple(m, v);
}

// Posterior (Cholesky)
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> GpM52Kernel::posterior_chol(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd m(xstar.rows(), amp_sq.size());
    std::vector<Eigen::MatrixXd> v;
    v.reserve(amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_m52(x, xstar, amp_sq(k), ls_sq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        /*
          computes:
          kzz - kzx * kxx_inv * kxz

          but references only the lower triangular half and returns
          a symmetrical view
        */
        v.push_back(
            (
                cov_m52(xstar, amp_sq(k), ls_sq.row(k), sigma_sq(k)).array() -
                ((kxxstar.transpose() * lxxs[k].solve(kxxstar))).array()
            )
            .matrix().selfadjointView<Eigen::Lower>().llt().matrixL()
        );
    }
    return std::make_tuple(m, v);
}

// Posterior marginals
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> GpM52Kernel::marginals(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd m(xstar.rows(), amp_sq.size());
    Eigen::MatrixXd v(xstar.rows(), amp_sq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_m52(x, xstar, amp_sq(k), ls_sq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        // also ensures posterior variance > delta
        v.col(k) = ((amp_sq(k) + sigma_sq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).matrix();
    }
    return std::make_tuple(m, v);
}

// Expectation and jacobian
Eigen::VectorXd GpM52Kernel::E(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::VectorXd m = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_m52(x, xstar, amp_sq(k), ls_sq.row(k));
        m.noalias() += (kxxstar.transpose() * axxs[k]);
    }
    return m / amp_sq.size();
}

std::tuple<double, Eigen::VectorXd> GpM52Kernel::E_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar)
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    double m = 0;
    Eigen::VectorXd jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_m52(xstar, x, amp_sq(k), ls_sq.row(k));
        m += kxstarx.dot(axxs[k]);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_m52(xstar, x, d, amp_sq(k), ls_sq.row(k));
            jac(d) -= dkxstarx_dxstar.dot(axxs[k]);
        }
    }
    return std::make_tuple(m / amp_sq.size(), jac / amp_sq.size());
}

// Pure Exploration objective and jacobian
Eigen::VectorXd GpM52Kernel::px(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd v = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_m52(x, xstar, amp_sq(k), ls_sq.row(k));
        v.noalias() -= ((amp_sq(k) + sigma_sq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).matrix();
    }
    return v / amp_sq.size();
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> GpM52Kernel::px_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar)
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x.rows(), x.rows());
    Eigen::RowVectorXd v = Eigen::VectorXd::Zero(1);
    Eigen::MatrixXd jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k<amp_sq.size(); k++){
        kxstarx = cov_1d_m52(xstar, x, amp_sq(k), ls_sq.row(k));
        v.noalias() -= ((amp_sq(k) + sigma_sq(k)) - ((kxstarx * lxxs[k].solve(kxstarx.transpose())).diagonal()).array()).cwiseMax(delta).matrix();
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_m52(xstar, x, d, amp_sq(k), ls_sq.row(k));
            jac(d) += (kxstarx * lxxs[k].solve(-dkxstarx_dxstar.transpose()));
        }
    }
    return std::make_tuple(v / amp_sq.size(), jac*2 / amp_sq.size());
}

// Expected improvement objective and jacobian
Eigen::VectorXd GpM52Kernel::ei(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ei = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::VectorXd gamma(xstar.rows());
    double ystar = y.minCoeff();
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_m52(x, xstar, amp_sq(k), ls_sq.row(k));
        mean = kxxstar.transpose() * axxs[k];
        sd = ((amp_sq(k) + sigma_sq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).sqrt().matrix();
        gamma = (ystar - mean.array()).array() / sd.array();
        ei.array() -= ((ystar - mean.array()) * ncdf(gamma).array() + sd.array() * npdf(gamma).array()).array();  
    }
    return ei / amp_sq.size();
}

std::tuple<double, Eigen::VectorXd> GpM52Kernel::ei_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar)
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
        kxstarx = cov_1d_m52(xstar, x, amp_sq(k), ls_sq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (amp_sq(k) + sigma_sq(k)) - kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        gamma = (ystar - mean) / sd;
        ei -= (ystar - mean) * ncdfd(gamma) + sd * npdfd(gamma);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_m52(xstar, x, d, amp_sq(k), ls_sq.row(k));
            sd_grad = -(kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose()))) / (sd);
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            gamma_grad = (((mean - ystar) * sd_grad) - (sd * mean_grad)) / var;
            jac(d) -= (ystar-mean) * gamma_grad * npdfd(gamma) - ncdfd(gamma) * mean_grad + npdfd(gamma)*sd_grad + sd * gamma_grad * npdfdg(gamma);
        }
    }
    return std::make_tuple(ei/amp_sq.size(), jac/amp_sq.size());
}

// upper confidence bound objective and jacobian
Eigen::VectorXd GpM52Kernel::ucb(Eigen::Ref<const Eigen::MatrixXd> xstar, double beta)
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ucb = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_m52(x, xstar, amp_sq(k), ls_sq.row(k));
        mean = kxxstar.transpose() * axxs[k];
        sd = ((amp_sq(k) + sigma_sq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).sqrt().matrix();
        ucb.array() += mean.array() - (sqrt(beta) * sd.array());
    }
    return ucb / amp_sq.size();
}

std::tuple<double, Eigen::VectorXd> GpM52Kernel::ucb_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar, double beta)
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
        kxstarx = cov_1d_m52(xstar, x, amp_sq(k), ls_sq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (amp_sq(k) + sigma_sq(k)) - kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        ucb += mean - (srbeta * sd);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_m52(xstar, x, d, amp_sq(k), ls_sq.row(k));
            sd_grad = -(kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose()))) / (sd);
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            jac(d) += mean_grad - (srbeta * sd_grad);
        }
    }
    return std::make_tuple(ucb/amp_sq.size(), jac/amp_sq.size());
}

// LOO_CV predictive density
Eigen::MatrixXd GpM52Kernel::loo_cv()
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
double GpM52Kernel::entropy(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    const double e = std::exp(1.0);
    double h = 0;
    double tmp = xstar.rows() / 2.0 *  log(e * M_PI * 2.0);
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::MatrixXd c(xstar.rows(), xstar.rows());
    for (int k=0; k<amp_sq.size(); k++){
        kxxstar = cross_cov_m52(x, xstar, amp_sq(k), ls_sq.row(k));
        /*
        computes

        h = tmp + 0.5 * 2 * log(det(Sigma))
          = tmp + 0.5 * 2 * log(det(kzz - kzx * kxx_inv * kxz))
          = tmp + sum(log(diag(chol(kzz - kzx * kxx_inv * kxz))))
  
        in a way that (hopefully) lets Eigen optimise the calculation
        */
        h += tmp + ( cov_m52(xstar, amp_sq(k), ls_sq.row(k), sigma_sq(k)).array() -
                   ((kxxstar.transpose() * lxxs[k].solve(kxxstar))
                   ).array()).matrix().llt().matrixLLT().diagonal().array().log().sum();

    }
    return h / amp_sq.size();
}
