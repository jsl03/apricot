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

#include "vanilla_expq.h".h"
#include "vanilla_matern52.h".h"

/* 
 Pybind -------------------------------------------------------------------------
*/ 

namespace py = pybind11;

/* MAKE SURE MODULE NAME MATCHES SETUP.PY*/
PYBIND11_MODULE(gp, m)
{
  m.doc() = "Internal apricot GP classes";
  m.def("npdf", &npdf);
  m.def("ncdf", &ncdf);
  m.def("cov_se", &cov_se);
  m.def("cross_cov_se", &cross_cov_se);
  m.def("cov_m52", &cov_m52);
  m.def("cross_cov_m52", &cross_cov_m52);
  m.def("cov_1d_m52", &cov_1d_m52);
  m.def("cross_cov_grad_m52", &cross_cov_grad_m52);

  py::class_<VanillaExpq>(m, "VanillaExpq")
    .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, double>())
    .def("view_x", &VanillaExpq::view_x)
    .def("view_y", &VanillaExpq::view_y)
    .def("view_parameters", &VanillaExpq::view_parameters)
    .def("view_lxx", &VanillaExpq::view_lxx)
    .def("marginals", &VanillaExpq::marginals)
    .def("posterior", &VanillaExpq::posterior)
    .def("posterior_chol", &VanillaExpq::posterior_chol)
    .def("E", &VanillaExpq::E)
    .def("E_jac", &VanillaExpq::E_jac)
    .def("px", &VanillaExpq::px)
    .def("px_jac", &VanillaExpq::px_jac)
    .def("ei", &VanillaExpq::ei)
    .def("ei_jac", &VanillaExpq::ei_jac)
    .def("ucb", &VanillaExpq::ucb)
    .def("ucb_jac", &VanillaExpq::ucb_jac)
    .def("loo_cv", &VanillaExpq::loo_cv)
    .def("entropy", &VanillaExpq::entropy)
    .def("__repr__",
         [](const VanillaExpq &a) {
           return "<apricot.core.VanillaExpq>";
         }
         );

  py::class_<VanillaM52>(m, "VanillaM52")
    .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, double>())
    .def("view_x", &VanillaM52::view_x)
    .def("view_y", &VanillaM52::view_y)
    .def("view_parameters", &VanillaM52::view_parameters)
    .def("view_lxx", &VanillaM52::view_lxx)
    .def("marginals", &VanillaM52::marginals)
    .def("posterior", &VanillaM52::posterior)
    .def("posterior_chol", &VanillaM52::posterior_chol)
    .def("E", &VanillaM52::E)
    .def("E_jac", &VanillaM52::E_jac)
    .def("px", &VanillaM52::px)
    .def("px_jac", &VanillaM52::px_jac)
    .def("ei", &VanillaM52::ei)
    .def("ei_jac", &VanillaM52::ei_jac)
    .def("ucb", &VanillaM52::ucb)
    .def("ucb_jac", &VanillaM52::ucb_jac)
    .def("loo_cv", &VanillaM52::loo_cv)
    .def("entropy", &VanillaM52::entropy)
    .def("__repr__",
         [](const VanillaM52 &a) {
           return "<apricot.core.VanillaM52>";
         }
         );
}
