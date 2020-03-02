/*
    This is not very pretty code.
    Proceed at your own risk.
    You have been warned.
*/

#define _USE_MATH_DEFINES

#include "kernels.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <math.h>
#include <unsupported/Eigen/SpecialFunctions>

// ----------------
// Class
// ----------------

class Gp
{
private:
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
    Eigen::VectorXd asq;
    Eigen::MatrixXd lsq;
    Eigen::VectorXd ssq;
    double delta;
    std::vector<Eigen::LLT<Eigen::MatrixXd>> lxxs;
    std::vector<Eigen::VectorXd> axxs;

public:
    Gp(
        const Eigen::MatrixXd &xdata,
        const Eigen::VectorXd &ydata,
        const Eigen::VectorXd &alpha,
        const Eigen::MatrixXd &rho,
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
Gp::Gp(
    const Eigen::MatrixXd &xdata,
    const Eigen::VectorXd &ydata,
    const Eigen::VectorXd &alpha,
    const Eigen::MatrixXd &rho,
    const Eigen::VectorXd &sigma,
    const double jitter):
        x(xdata),
        y(ydata),
        asq(alpha),
        lsq(rho),
        ssq(sigma),
        delta(jitter)
{
    for (int k=0; k<asq.size(); k++){
        lxxs.reserve(asq.size());
        axxs.reserve(asq.size());
        lxxs.push_back(L_cov_se(x, asq(k), lsq.row(k), ssq(k)));
        axxs.push_back(lxxs[k].solve(y));
    }
}


// provide a view of x data
const Eigen::MatrixXd Gp::view_x()
{
    return x;
}

// provide a view of y data
const Eigen::VectorXd Gp::view_y()
{
    return y;
}

// provide a view of parameters
std::tuple<const Eigen::VectorXd, const Eigen::MatrixXd, const Eigen::VectorXd> Gp::view_parameters()
{
    return std::make_tuple(asq, lsq, ssq);
}

// provide a view of covariance matrix cholesky factors
std::vector<Eigen::MatrixXd> Gp::view_lxx()
{
    std::vector<Eigen::MatrixXd> lxx_views;
    lxx_views.reserve(asq.size());
    for (int k=0; k<asq.size(); k++){
        lxx_views.push_back(lxxs[k].matrixL());
    }
    return lxx_views;
}

// Posterior
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> Gp::posterior(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd m(xstar.rows(), asq.size());
    std::vector<Eigen::MatrixXd> v;
    v.reserve(asq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<asq.size(); k++){
        kxxstar = cross_cov_se(x, xstar, asq(k), lsq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        /*
        computes:
        kzz - kzx * kxx_inv * kxz

        but references only the lower triangular half and returns
        a symmetrical view
        */
        v.push_back(
             (
                 cov_se(xstar, asq(k), lsq.row(k), ssq(k)).array() -
                 ((kxxstar.transpose() * lxxs[k].solve(kxxstar))).array()
             )
             .matrix().selfadjointView<Eigen::Lower>()
        );
    }
    return std::make_tuple(m, v);
}

// Posterior (Cholesky)
std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>> Gp::posterior_chol(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd m(xstar.rows(), asq.size());
    std::vector<Eigen::MatrixXd> v;
    v.reserve(asq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<asq.size(); k++){
        kxxstar = cross_cov_se(x, xstar, asq(k), lsq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        /*
          computes:
          kzz - kzx * kxx_inv * kxz

          but references only the lower triangular half and returns
          a symmetrical view
        */
        v.push_back(
            (
                cov_se(xstar, asq(k), lsq.row(k), ssq(k)).array() -
                ((kxxstar.transpose() * lxxs[k].solve(kxxstar))).array()
            )
            .matrix().selfadjointView<Eigen::Lower>().llt().matrixL()
        );
    }
    return std::make_tuple(m, v);
}

// Posterior marginals
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> Gp::marginals(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd m(xstar.rows(), asq.size());
    Eigen::MatrixXd v(xstar.rows(), asq.size());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<asq.size(); k++){
        kxxstar = cross_cov_se(x, xstar, asq(k), lsq.row(k));
        m.col(k) = kxxstar.transpose() * axxs[k];
        // also ensures posterior variance > delta
        v.col(k) = ((asq(k) + ssq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).matrix();
    }
    return std::make_tuple(m, v);
}

// Expectation and jacobian
Eigen::VectorXd Gp::E(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::VectorXd m = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    for (int k=0; k<asq.size(); k++){
        kxxstar = cross_cov_se(x, xstar, asq(k), lsq.row(k));
        m.noalias() += (kxxstar.transpose() * axxs[k]);
    }
    return m / asq.size();
}

std::tuple<double, Eigen::VectorXd> Gp::E_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar)
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    double m = 0;
    Eigen::VectorXd jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k<asq.size(); k++){
        kxstarx = cov_1d_se(xstar, x, asq(k), lsq.row(k));
        m += kxstarx.dot(axxs[k]);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_se(xstar, x, d, asq(k), lsq.row(k), kxstarx);
            jac(d) -= dkxstarx_dxstar.dot(axxs[k]);
        }
    }
    return std::make_tuple(m / asq.size(), jac / asq.size());
}

// Pure Exploration objective and jacobian
Eigen::VectorXd Gp::px(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd v = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k<asq.size(); k++){
        kxxstar = cross_cov_se(x, xstar, asq(k), lsq.row(k));
        v.noalias() -= ((asq(k) + ssq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).matrix();
    }
    return v / asq.size();
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> Gp::px_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar)
{
    Eigen::RowVectorXd kxstarx(x.rows());
    Eigen::RowVectorXd dkxstarx_dxstar(x.rows());
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x.rows(), x.rows());
    Eigen::RowVectorXd v = Eigen::VectorXd::Zero(1);
    Eigen::MatrixXd jac = Eigen::VectorXd::Zero(xstar.size());
    for (int k=0; k<asq.size(); k++){
        kxstarx = cov_1d_se(xstar, x, asq(k), lsq.row(k));
        v.noalias() -= ((asq(k) + ssq(k)) - ((kxstarx * lxxs[k].solve(kxstarx.transpose())).diagonal()).array()).cwiseMax(delta).matrix();
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_se(xstar, x, d, asq(k), lsq.row(k), kxstarx);
            jac(d) += (kxstarx * lxxs[k].solve(-dkxstarx_dxstar.transpose()));
        }
    }
    return std::make_tuple(v / asq.size(), jac*2 / asq.size());
}

// Expected improvement objective and jacobian
Eigen::VectorXd Gp::ei(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ei = Eigen::VectorXd::Zero(xstar.rows());
    Eigen::VectorXd gamma(xstar.rows());
    double ystar = y.minCoeff();
    for (int k=0; k<asq.size(); k++){
        kxxstar = cross_cov_se(x, xstar, asq(k), lsq.row(k));
        mean = kxxstar.transpose() * axxs[k];
        sd = ((asq(k) + ssq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).sqrt().matrix();
        gamma = (ystar - mean.array()).array() / sd.array();
        ei.array() -= ((ystar - mean.array()) * ncdf(gamma).array() + sd.array() * npdf(gamma).array()).array();  
    }
    return ei / asq.size();
}

std::tuple<double, Eigen::VectorXd> Gp::ei_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar)
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
    for (int k=0; k<asq.size(); k++){
        kxstarx = cov_1d_se(xstar, x, asq(k), lsq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (asq(k) + ssq(k)) - kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        gamma = (ystar - mean) / sd;
        ei -= (ystar - mean) * ncdfd(gamma) + sd * npdfd(gamma);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_se(xstar, x, d, asq(k), lsq.row(k), kxstarx);
            sd_grad = -(kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose()))) / (sd);
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            gamma_grad = (((mean - ystar) * sd_grad) - (sd * mean_grad)) / var;
            jac(d) -= (ystar-mean) * gamma_grad * npdfd(gamma) - ncdfd(gamma) * mean_grad + npdfd(gamma)*sd_grad + sd * gamma_grad * npdfdg(gamma);
        }
    }
    return std::make_tuple(ei/asq.size(), jac/asq.size());
}

// upper confidence bound objective and jacobian
Eigen::VectorXd Gp::ucb(Eigen::Ref<const Eigen::MatrixXd> xstar, double beta)
{
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::VectorXd mean(xstar.rows());
    Eigen::VectorXd sd(xstar.rows());
    Eigen::VectorXd ucb = Eigen::VectorXd::Zero(xstar.rows());
    for (int k=0; k<asq.size(); k++){
        kxxstar = cross_cov_se(x, xstar, asq(k), lsq.row(k));
        mean = kxxstar.transpose() * axxs[k];
        sd = ((asq(k) + ssq(k)) - ((kxxstar.transpose() * lxxs[k].solve(kxxstar)).diagonal()).array()).cwiseMax(delta).sqrt().matrix();
        ucb.array() += mean.array() - (sqrt(beta) * sd.array());
    }
    return ucb / asq.size();
}

std::tuple<double, Eigen::VectorXd> Gp::ucb_jac(Eigen::Ref<const Eigen::RowVectorXd> xstar, double beta)
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
    for (int k=0; k<asq.size(); k++){
        kxstarx = cov_1d_se(xstar, x, asq(k), lsq.row(k));
        mean = kxstarx.dot(axxs[k]);
        var = (asq(k) + ssq(k)) - kxstarx.dot(lxxs[k].solve(kxstarx.transpose()));
        sd = sqrt(var);
        ucb += mean - (srbeta * sd);
        for (int d=0; d<xstar.size(); d++){
            dkxstarx_dxstar = cross_cov_grad_se(xstar, x, d, asq(k), lsq.row(k), kxstarx);
            sd_grad = -(kxstarx.dot(lxxs[k].solve(-dkxstarx_dxstar.transpose()))) / (sd);
            mean_grad = -dkxstarx_dxstar.dot(axxs[k]);
            jac(d) += mean_grad - (srbeta * sd_grad);
        }
    }
    return std::make_tuple(ucb/asq.size(), jac/asq.size());
}

// LOO_CV predictive density
Eigen::MatrixXd Gp::loo_cv()
{
    Eigen::MatrixXd cv(asq.rows(), x.rows());
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x.rows(), x.rows());
    Eigen::VectorXd c_diag;
    for (int k=0; k<asq.size(); k++){
        c_diag = lxxs[k].solve(I).diagonal();
        cv.row(k) = -0.5*log(2*M_PI) + (0.5*(c_diag.array().log())) - 0.5 * (axxs[k].array().square() / c_diag.array());
    }
    return cv;
}

// Entropy
double Gp::entropy(Eigen::Ref<const Eigen::MatrixXd> xstar)
{
    const double e = std::exp(1.0);
    double h = 0;
    double tmp = xstar.rows() / 2.0 *  log(e * M_PI * 2.0);
    Eigen::MatrixXd kxxstar(x.rows(), xstar.rows());
    Eigen::MatrixXd c(xstar.rows(), xstar.rows());
    for (int k=0; k<asq.size(); k++){
        kxxstar = cross_cov_se(x, xstar, asq(k), lsq.row(k));
        /*
        computes

        h = tmp + 0.5 * 2 * log(det(Sigma))
          = tmp + 0.5 * 2 * log(det(kzz - kzx * kxx_inv * kxz))
          = tmp + sum(log(diag(chol(kzz - kzx * kxx_inv * kxz))))
  
        in a way that (hopefully) lets Eigen optimise the calculation
        */
        h += tmp + ( cov_se(xstar, asq(k), lsq.row(k), ssq(k)).array() -
                   ((kxxstar.transpose() * lxxs[k].solve(kxxstar))
                   ).array()).matrix().llt().matrixLLT().diagonal().array().log().sum();

    }
    return h / asq.size();
}


// ----------------
// Pybind
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(gp, m)
{
  m.doc() = "pybind11 example plugin";
  m.def("npdf", &npdf);
  m.def("ncdf", &ncdf);
  m.def("cov_se", &cov_se);
  m.def("cross_cov_se", &cross_cov_se);
  m.def("cov_m52", &cov_m52);
  m.def("cross_cov_m52", &cross_cov_m52);

  py::class_<Gp>(m, "Gp")
    .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, double>())
    .def("view_x", &Gp::view_x)
    .def("view_y", &Gp::view_y)
    .def("view_parameters", &Gp::view_parameters)
    .def("view_lxx", &Gp::view_lxx)
    .def("marginals", &Gp::marginals)
    .def("posterior", &Gp::posterior)
    .def("posterior_chol", &Gp::posterior_chol)
    .def("E", &Gp::E)
    .def("E_jac", &Gp::E_jac)
    .def("px", &Gp::px)
    .def("px_jac", &Gp::px_jac)
    .def("ei", &Gp::ei)
    .def("ei_jac", &Gp::ei_jac)
    .def("ucb", &Gp::ucb)
    .def("ucb_jac", &Gp::ucb_jac)
    .def("loo_cv", &Gp::loo_cv)
    .def("entropy", &Gp::entropy)
    .def("__repr__",
         [](const Gp &a) {
           return "<example.Gp>";
         }
         );
}
