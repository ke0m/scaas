/**
 * Python interface to the half-order integration/
 * differentation operator
 * @author: Joseph Jennings
 * @version: 2020.03.18
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "halfint.h"

namespace py = pybind11;

PYBIND11_MODULE(halfint,m) {
  m.doc() = "Half-order integration/differentation operator";

  py::class_<halfint>(m,"halfint")
      .def(py::init<bool,int,float>(),
          py::arg("inv"), py::arg("n1"), py::arg("rho"))
      .def("forward",[](halfint &hfi,
          bool add,
          int n1,
          py::array_t<float, py::array::c_style> mod,
          py::array_t<float, py::array::c_style> dat)
          {
            hfi.forward(add,n1,mod.mutable_data(),dat.mutable_data());
          },
          py::arg("add"), py::arg("n1"), py::arg("mod"), py::arg("dat")
          )
      .def("adjoint",[](halfint &hfi,
          bool add,
          int n1,
          py::array_t<float, py::array::c_style> mod,
          py::array_t<float, py::array::c_style> dat)
          {
            hfi.adjoint(add,n1,mod.mutable_data(),dat.mutable_data());
          },
          py::arg("add"), py::arg("n1"), py::arg("mod"), py::arg("dat")
          );
}
