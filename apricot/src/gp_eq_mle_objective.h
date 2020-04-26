// This file is licensed under Version 3.0 of the GNU General Public
// License. See LICENSE for a text of the license.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// THIS IS AN EXPERIMENTAL MODULE
// -----------------------------------------------------------------------------

double halfLogDet(const Eigen::MatrixXd &mat)
{
        double halflogdet = 0;
        for (int i=0; i < mat.rows(); i++){
                halflogdet += log(mat(i,i));
        }
        return halflogdet;
}

Eigen::MatrixXd covEq(
        Eigen::Ref<const Eigen::MatrixXd> x,
        double amp_sq,
        Eigen::Ref<const Eigen::VectorXd> ls_sq,
        double sigma_sq
        )
{
        Eigen::RowVectorXd ls = ls_sq.transpose().array().sqrt();
        Eigen::MatrixXd tau(x.rows(), x.rows());
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x.rows(), x.rows());
        Eigen::MatrixXd xl = x.array().rowwise() / ls.array();
        Eigen::VectorXd tmp = xl.array().square().rowwise().sum();
        // next is an outer addition of two vectors:
        tau = tmp.rowwise().replicate(tmp.size()) +
                tmp.transpose().colwise().replicate(tmp.size());
        tau.noalias() += -2.0 * xl * xl.transpose();
        return ((amp_sq * ((-0.5 * tau).array().exp())).array()
                + (I.array() * sigma_sq)
                ).matrix().selfadjointView<Eigen::Lower>();
}

class make_mle_objective
{
private:
        Eigen::MatrixXd x;
        Eigen::VectorXd y;
        double jitter;
        double term0;

public:
        make_mle_objective(
                const Eigen::MatrixXd &x,
                const Eigen::VectorXd &y,
                const double jitter
                );

        double objective(Eigen::Ref<const Eigen::VectorXd> log_theta);
};

make_mle_objective::make_mle_objective(
        const Eigen::MatrixXd &x,
        const Eigen::VectorXd &y,
        const double jitter
        ):
        x(x),
        y(y),
        jitter(jitter)
{
        term0 = (x.rows() / 2.0) * log(2.0 * M_PI);
}

double make_mle_objective::objective(
        Eigen::Ref<const Eigen::VectorXd> log_theta
        )
{
    // retrieve from log-space
    Eigen::VectorXd theta = log_theta.array().exp();
    double amp_sq = pow(theta(0), 2);
    double sigma_sq = pow(theta(1), 2);
    Eigen::VectorXd ls_sq = theta.tail(theta.size() - 2).array().square();
    Eigen::MatrixXd knn(x.rows(), x.rows());
    Eigen::LLT<Eigen::MatrixXd> k_chol;
    double term1;
    double term2;
    // no sigma_sq yet! (just jitter)
    knn = covEq(x, amp_sq, ls_sq, sigma_sq + jitter);
    k_chol = knn.selfadjointView<Eigen::Lower>().llt();
    // log determinant term
    term1 = halfLogDet(k_chol.matrixL());
    // likelihood term
    term2 = 0.5 * k_chol.matrixL().solve(y).squaredNorm();
    return (term0 + term1 + term2) / y.size();
}
