/**
 * Python interface for converting offset to angle
 * @author: Joseph Jennings
 * @version: 2020.03.22
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "adcig.h"

namespace py = pybind11;

PYBIND11_MODULE(adcig,m) {
  m.doc() = "Conversion of subsurface offset images to angle";
  m.def("convert2ang",[](
      int nro,
      int nx,
      int nh,
      float oh,
      float dh,
      int nta,
      float ota,
      float dta,
      int na,
      float oa,
      float da,
      int nz,
      float oz,
      float dz,
      int ext,
      py::array_t<float, py::array::c_style> off,
      py::array_t<float, py::array::c_style> ang
      )
      {
        convert2ang(nro, nx, nh, oh, dh, nta, ota, dta, na, oa ,da, nz, oz, dz,
            ext, off.mutable_data(), ang.mutable_data());
      },
      py::arg("nro"), py::arg("nx"), py::arg("nh"), py::arg("oh"), py::arg("dh"),
      py::arg("nta"), py::arg("ota"), py::arg("dta"), py::arg("na"), py::arg("oa"), py::arg("da"),
      py::arg("nz"), py::arg("oz"), py::arg("dz"), py::arg("ext"), py::arg("off"), py::arg("ang")
      );
}
