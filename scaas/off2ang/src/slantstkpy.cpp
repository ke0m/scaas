/**
 * Python interface to the slant stack operator
 * @author: Joseph Jennings
 * @version: 2020.03.19
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "slantstk.h"

namespace py = pybind11;

PYBIND11_MODULE(slantstk,m) {
  m.doc() = "Slant-stack operator";

  py::class_<slantstk>(m,"slantstk")
      .def(py::init<bool,int,float,float,int,float,float,int,float,float,float,float>(),
           py::arg("rho"), py::arg("nx"), py::arg("ox"), py::arg("dx"),
           py::arg("ns"), py::arg("os"), py::arg("ds"),
           py::arg("nt"), py::arg("ot"), py::arg("dt"), py::arg("s11"), py::arg("anti1"))
      .def("forward",[](slantstk &ssk,
          bool add,
          int nm,
          int nd,
          py::array_t<float, py::array::c_style> mod,
          py::array_t<float, py::array::c_style> dat)
          {
            ssk.forward(add, nm, nd, mod.mutable_data(), dat.mutable_data());
          },
          py::arg("add"), py::arg("nm"), py::arg("nd"),
          py::arg("mod"), py::arg("dat")
          )
      .def("adjoint",[](slantstk &ssk,
          bool add,
          int nm,
          int nd,
          py::array_t<float, py::array::c_style> mod,
          py::array_t<float, py::array::c_style> dat)
          {
            ssk.adjoint(add, nm, nd, mod.mutable_data(), dat.mutable_data());
          },
          py::arg("add"), py::arg("nm"), py::arg("nd"),
          py::arg("mod"), py::arg("dat")
          );
}

