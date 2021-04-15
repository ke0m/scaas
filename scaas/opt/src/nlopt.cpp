/**
 * Python interface to Huy's solvers
 * @author: Joseph Jennings
 * @version: 2019.11.21
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "lbfgs.h"

namespace py = pybind11;

void lbfgs_pyf(size_t n, int* m, float *x, float *f, float *g, bool diagco, float *diag, float *w, int *iflag, int *isave, float *dsave) {
  int m0 = m[0]; float f0 = f[0]; int iflag0 = iflag[0];
  lbfgs(n, m0, x, f0, g, diagco, diag, w, iflag0, isave, dsave);
  m[0] = m0; f[0] = f0; iflag[0] = iflag0;
}

void nlcg_pyf(size_t n, float *x, float *f, float *g, float *diag, float *w, int *iflag, int *isave, float *dsave) {
  float f0 = f[0]; int iflag0 = iflag[0];
  nlcg(n, x, f0, g, diag, w, iflag0, isave, dsave);
  f[0] = f0; iflag[0] = iflag0;
}

void steepest_pyf(size_t n, float *x, float *f, float *g, float *diag, float *w, int *iflag, int *isave, float *dsave) {
  float f0 = f[0]; int iflag0 = iflag[0];
  steepest(n, x, f0, g, diag, w, iflag0, isave, dsave);
  f[0] = f0; iflag[0] = iflag0;
}

PYBIND11_MODULE(nlopt,m) {
  m.doc() = "Non-linear optimization functions";
  m.def("lbfgs",[](
          py::size_t n,
          py::array_t<int, py::array::c_style> m,
          py::array_t<float, py::array::c_style> x,
          py::array_t<float, py::array::c_style> f,
          py::array_t<float, py::array::c_style> g,
          bool diagco,
          py::array_t<float, py::array::c_style> diag,
          py::array_t<float, py::array::c_style> w,
          py::array_t<int, py::array::c_style> iflag,
          py::array_t<int, py::array::c_style> isave,
          py::array_t<float, py::array::c_style> dsave)
          {
            lbfgs_pyf(n, m.mutable_data(), x.mutable_data(), f.mutable_data(), g.mutable_data(),
                diagco, diag.mutable_data(), w.mutable_data(), iflag.mutable_data(),
                isave.mutable_data(), dsave.mutable_data());
          },
          py::arg("n"), py::arg("m"), py::arg("x"), py::arg("f"), py::arg("g"),
          py::arg("diagco"), py::arg("diag"), py::arg("w"), py::arg("iflag"),
          py::arg("isave"), py::arg("dsave")
       );
   m.def("nlcg",[](
          py::size_t n,
          py::array_t<float, py::array::c_style> x,
          py::array_t<float, py::array::c_style> f,
          py::array_t<float, py::array::c_style> g,
          py::array_t<float, py::array::c_style> diag,
          py::array_t<float, py::array::c_style> w,
          py::array_t<int, py::array::c_style> iflag,
          py::array_t<int, py::array::c_style> isave,
          py::array_t<float, py::array::c_style> dsave)
          {
            nlcg_pyf(n, x.mutable_data(), f.mutable_data(), g.mutable_data(),
                diag.mutable_data(), w.mutable_data(), iflag.mutable_data(),
                isave.mutable_data(), dsave.mutable_data());
          },
          py::arg("n"), py::arg("x"), py::arg("f"), py::arg("g"),
          py::arg("diag"), py::arg("w"), py::arg("iflag"),
          py::arg("isave"), py::arg("dsave")
       );
   m.def("steepest",[](
          py::size_t n,
          py::array_t<float, py::array::c_style> x,
          py::array_t<float, py::array::c_style> f,
          py::array_t<float, py::array::c_style> g,
          py::array_t<float, py::array::c_style> diag,
          py::array_t<float, py::array::c_style> w,
          py::array_t<int, py::array::c_style> iflag,
          py::array_t<int, py::array::c_style> isave,
          py::array_t<float, py::array::c_style> dsave)
          {
           steepest_pyf(n, x.mutable_data(), f.mutable_data(), g.mutable_data(),
                diag.mutable_data(), w.mutable_data(), iflag.mutable_data(),
                isave.mutable_data(), dsave.mutable_data());
          },
          py::arg("n"), py::arg("x"), py::arg("f"), py::arg("g"),
          py::arg("diag"), py::arg("w"), py::arg("iflag"),
          py::arg("isave"), py::arg("dsave")
        );
}
