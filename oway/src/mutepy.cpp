/**
 * Python interface for muting function
 * @author: Joseph Jennings
 * @version: 2020.07.11
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mute.h"

namespace py = pybind11;

PYBIND11_MODULE(mutter,m) {
  m.doc() = "Applies a mute to seismic gathers";
  m.def("muteall",[](
      int n3,
      int n2, float o2, float d2,
      int n1, float o1, float d1,
      float tp, float t0, float v0,
      float slope0, float slopep, float x0,
      bool abs, bool inner, bool hyper, bool half,
      py::array_t<float, py::array::c_style> datin,
      py::array_t<float, py::array::c_style> datot
      )
      {
        muteall(n3,n2,o2,d2,n1,o1,d1,
                tp,t0,v0,slope0,slopep,x0,
                abs,inner,hyper,half,
                datin.mutable_data(),datot.mutable_data());
      },
      py::arg("n3"), py::arg("n2"), py::arg("o2"), py::arg("d2"), py::arg("n1"), py::arg("o1"), py::arg("d1"),
      py::arg("tp"), py::arg("t0"), py::arg("v0"), py::arg("slope0"), py::arg("slopep"), py::arg("x0"),
      py::arg("abs"), py::arg("inner"), py::arg("hyper"), py::arg("half"),
      py::arg("datin"), py::arg("datot")
      );
}
