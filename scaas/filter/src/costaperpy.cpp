/**
 * Python interface for cosine taper function
 * @author: Joseph Jennings
 * @version: 2020.07.17
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "costaper.h"

namespace py = pybind11;

PYBIND11_MODULE(costapkern,m) {
  m.doc() = "Applies a mute to seismic gathers";
  m.def("costapkern",[](
      int dim, int dim1,
      int n1, int n2,
      py::array_t<int, py::array::c_style> n,
      py::array_t<int, py::array::c_style> nw,
      py::array_t<int, py::array::c_style> s,
      py::array_t<float, py::array::c_style> data
      )
      {
        costaper(dim,dim1,n1,n2,
                 n.mutable_data(),nw.mutable_data(),s.mutable_data(),
                 data.mutable_data());
      },
      py::arg("dim"), py::arg("dim1"), py::arg("n1"), py::arg("n2"),
      py::arg("n"),   py::arg("nw"), py::arg("s"),
      py::arg("data")
      );
}
