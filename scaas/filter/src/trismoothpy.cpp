/**
 * Python interface to a triangular smoother
 * @author: Joseph Jennings
 * @version: 2020.03.18
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "trismooth.h"

namespace py = pybind11;

PYBIND11_MODULE(trismoothkern,m) {
  m.doc() = "Triangular smoother that works in n dimensions";
  m.def("smooth2",[](
      int dim1,
      int n1,
      int n2,
      py::array_t<int, py::array::c_style> n,
      py::array_t<int, py::array::c_style> rect,
      py::array_t<int, py::array::c_style> s,
      py::array_t<float, py::array::c_style> data
      )
      {
        smooth2(dim1, n1, n2, n.mutable_data(), rect.mutable_data(), s.mutable_data(), data.mutable_data());
      },
      py::arg("dim1"), py::arg("n1"), py::arg("n2"),
      py::arg("n"), py::arg("rect"), py::arg("s"), py::arg("data")
      );
}
