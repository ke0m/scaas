/**
 * Python interface for fourier-domain based off2ang
 * @author: Joseph Jennings
 * @version: 2020.08.07
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "adcigkzkx.h"

namespace py = pybind11;

PYBIND11_MODULE(adcigkzkx,m) {
  m.doc() = "Conversion of subsurface offset images to angle";
  m.def("convert2angkzkykx",[](
      int ngat,
      int nz, float oz, float dz,
      int nhy, float ohy, float dhy,
      int nhx, float ohx, float dhx,
      float oa, float da,
      py::array_t<std::complex<float>, py::array::c_style> off,
      py::array_t<std::complex<float>, py::array::c_style> ang,
      float eps, int nthrd, bool verb
      )
      {
        convert2angkzkykx(ngat, nz, oz, dz, nhy, ohy, dhy, nhx, ohx ,dhx, oa, da,
            off.mutable_data(), ang.mutable_data(), eps, nthrd, verb);
      },
      py::arg("ngat"), py::arg("nz"), py::arg("oz"), py::arg("dz"),
      py::arg("nhy"), py::arg("ohy"), py::arg("dhy"), py::arg("nhx"), py::arg("ohx"), py::arg("dhx"),
      py::arg("oa"), py::arg("da"), py::arg("off"), py::arg("ang"), py::arg("eps"),
      py::arg("nthrd"), py::arg("verb")
      );
}
