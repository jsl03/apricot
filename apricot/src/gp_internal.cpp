#define _USE_MATH_DEFINES

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <math.h>
#include <unsupported/Eigen/SpecialFunctions>

#include "misc.h"
#include "gp_eq_kernel.h"
#include "gp_m52_kernel.h"
#include "gp_m32_kernel.h"
#include "gp_rq_kernel.h"

/* Pybind */ 

namespace py = pybind11;

// module name must match that inside setup.py
PYBIND11_MODULE(gp_internal, m)
{
  m.doc() = "Internal apricot GP classes";
  m.def("npdf", &npdf);
  m.def("ncdf", &ncdf);
  m.def("cov_eq", &cov_eq);
  m.def("cross_cov_eq", &cross_cov_eq);
  m.def("cov_m52", &cov_m52);
  m.def("cross_cov_m52", &cross_cov_m52);
  m.def("cov_m32", &cov_m32);
  m.def("cross_cov_352", &cross_cov_m32);
  m.def("cov_rq", &cov_rq);
  m.def("cross_cov_rq", &cross_cov_rq);

  py::class_<GpEqKernel>(m, "GpEqKernel")
    .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, double>())
    .def("view_x", &GpEqKernel::view_x)
    .def("view_y", &GpEqKernel::view_y)
    .def("view_parameters", &GpEqKernel::view_parameters)
    .def("view_lxx", &GpEqKernel::view_lxx)
    .def("marginals", &GpEqKernel::marginals)
    .def("posterior", &GpEqKernel::posterior)
    .def("posterior_chol", &GpEqKernel::posterior_chol)
    .def("E", &GpEqKernel::E)
    .def("E_jac", &GpEqKernel::E_jac)
    .def("px", &GpEqKernel::px)
    .def("px_jac", &GpEqKernel::px_jac)
    .def("ei", &GpEqKernel::ei)
    .def("ei_jac", &GpEqKernel::ei_jac)
    .def("ucb", &GpEqKernel::ucb)
    .def("ucb_jac", &GpEqKernel::ucb_jac)
    .def("loo_cv", &GpEqKernel::loo_cv)
    .def("entropy", &GpEqKernel::entropy)
    .def("__repr__",
         [](const GpEqKernel &a) {
           return "<apricot.core.GpEqKernel>";
         }
         );

  py::class_<GpM52Kernel>(m, "GpM52Kernel")
    .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, double>())
    .def("view_x", &GpM52Kernel::view_x)
    .def("view_y", &GpM52Kernel::view_y)
    .def("view_parameters", &GpM52Kernel::view_parameters)
    .def("view_lxx", &GpM52Kernel::view_lxx)
    .def("marginals", &GpM52Kernel::marginals)
    .def("posterior", &GpM52Kernel::posterior)
    .def("posterior_chol", &GpM52Kernel::posterior_chol)
    .def("E", &GpM52Kernel::E)
    .def("E_jac", &GpM52Kernel::E_jac)
    .def("px", &GpM52Kernel::px)
    .def("px_jac", &GpM52Kernel::px_jac)
    .def("ei", &GpM52Kernel::ei)
    .def("ei_jac", &GpM52Kernel::ei_jac)
    .def("ucb", &GpM52Kernel::ucb)
    .def("ucb_jac", &GpM52Kernel::ucb_jac)
    .def("loo_cv", &GpM52Kernel::loo_cv)
    .def("entropy", &GpM52Kernel::entropy)
    .def("__repr__",
         [](const GpM52Kernel &a) {
           return "<apricot.core.GpM52Kernel>";
         }
         );

  py::class_<GpM32Kernel>(m, "GpM32Kernel")
    .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, double>())
    .def("view_x", &GpM32Kernel::view_x)
    .def("view_y", &GpM32Kernel::view_y)
    .def("view_parameters", &GpM32Kernel::view_parameters)
    .def("view_lxx", &GpM32Kernel::view_lxx)
    .def("marginals", &GpM32Kernel::marginals)
    .def("posterior", &GpM32Kernel::posterior)
    .def("posterior_chol", &GpM32Kernel::posterior_chol)
    .def("E", &GpM32Kernel::E)
    .def("E_jac", &GpM32Kernel::E_jac)
    .def("px", &GpM32Kernel::px)
    .def("px_jac", &GpM32Kernel::px_jac)
    .def("ei", &GpM32Kernel::ei)
    .def("ei_jac", &GpM32Kernel::ei_jac)
    .def("ucb", &GpM32Kernel::ucb)
    .def("ucb_jac", &GpM32Kernel::ucb_jac)
    .def("loo_cv", &GpM32Kernel::loo_cv)
    .def("entropy", &GpM32Kernel::entropy)
    .def("__repr__",
         [](const GpM32Kernel &a) {
           return "<apricot.core.GpM32Kernel>";
         }
         );

  py::class_<GpRqKernel>(m, "GpRqKernel")
    .def(py::init<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, double>())
    .def("view_x", &GpRqKernel::view_x)
    .def("view_y", &GpRqKernel::view_y)
    .def("view_parameters", &GpRqKernel::view_parameters)
    .def("view_lxx", &GpRqKernel::view_lxx)
    .def("marginals", &GpRqKernel::marginals)
    .def("posterior", &GpRqKernel::posterior)
    .def("posterior_chol", &GpRqKernel::posterior_chol)
    .def("E", &GpRqKernel::E)
    .def("E_jac", &GpRqKernel::E_jac)
    .def("px", &GpRqKernel::px)
    .def("px_jac", &GpRqKernel::px_jac)
    .def("ei", &GpRqKernel::ei)
    .def("ei_jac", &GpRqKernel::ei_jac)
    .def("ucb", &GpRqKernel::ucb)
    .def("ucb_jac", &GpRqKernel::ucb_jac)
    .def("loo_cv", &GpRqKernel::loo_cv)
    .def("entropy", &GpRqKernel::entropy)
    .def("__repr__",
         [](const GpRqKernel &a) {
           return "<apricot.core.GpRqKernel>";
         }
         );
}
