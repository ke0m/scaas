/**
 * Python interface for ssr3 wrapper function
 * @author: Joseph Jennings
 * @version: 2020.07.26
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "ssr3wrap.h"

namespace py = pybind11;

PYBIND11_MODULE(ssr3wrap,m) {
  m.doc() = "Wrapper for ssr3 ssf functions";
  m.def("ssr3modshots",[](
      int nx, int ny, int nz,
      float ox, float oy, float oz,
      float dx, float dy, float dz,
      int nw, float ow, float dw,
      int ntx, int nty, int padx, int pady,
      float dtmax, int nrmax,
      py::array_t<float, py::array::c_style> slo,
      int nexp,
      py::array_t<int, py::array::c_style> nsrc,
      py::array_t<float, py::array::c_style> srcy, py::array_t<float, py::array::c_style> srcx,
      py::array_t<int, py::array::c_style> nrec,
      py::array_t<float, py::array::c_style> recy, py::array_t<float, py::array::c_style> recx,
      py::array_t<std::complex<float>, py::array::c_style> wav,
      py::array_t<float, py::array::c_style> ref,
      py::array_t<std::complex<float>, py::array::c_style> dat,
      int nthrds, int verb
      )
      {
        ssr3_modshots(nx, ny, nz,
                      ox, oy, oz,
                      dx, dy, dz,
                      nw, ow, dw,
                      ntx, nty, padx, pady,
                      dtmax, nrmax,
                      slo.mutable_data(),
                      nexp,
                      nsrc.mutable_data(), srcy.mutable_data(), srcx.mutable_data(),
                      nrec.mutable_data(), recy.mutable_data(), recx.mutable_data(),
                      wav.mutable_data(), ref.mutable_data(), dat.mutable_data(),
                      nthrds, verb);
      },
      py::arg("nx"), py::arg("ny"), py::arg("nz"),
      py::arg("ox"), py::arg("oy"), py::arg("oz"),
      py::arg("dx"), py::arg("dy"), py::arg("dz"),
      py::arg("nw"), py::arg("ow"), py::arg("dw"),
      py::arg("ntx"), py::arg("nty"), py::arg("px"), py::arg("py"),
      py::arg("dtmax"), py::arg("nrmax"), py::arg("slo"),
      py::arg("nexp"),
      py::arg("nsrc"), py::arg("srcy"), py::arg("srcx"),
      py::arg("nrec"), py::arg("recy"), py::arg("recx"),
      py::arg("wav"), py::arg("ref"), py::arg("dat"),
      py::arg("nthrds"), py::arg("verb")
      );
  m.def("ssr3migshots",[](
      int nx, int ny, int nz,
      float ox, float oy, float oz,
      float dx, float dy, float dz,
      int nw, float ow, float dw,
      int ntx, int nty, int padx, int pady,
      float dtmax, int nrmax,
      py::array_t<float, py::array::c_style> slo,
      int nexp,
      py::array_t<int, py::array::c_style> nsrc,
      py::array_t<float, py::array::c_style> srcy, py::array_t<float, py::array::c_style> srcx,
      py::array_t<int, py::array::c_style> nrec,
      py::array_t<float, py::array::c_style> recy, py::array_t<float, py::array::c_style> recx,
      py::array_t<std::complex<float>, py::array::c_style> dat,
      py::array_t<std::complex<float>, py::array::c_style> wav,
      py::array_t<float, py::array::c_style> img,
      int nthrds, int verb
      )
      {
        ssr3_migshots(nx, ny, nz,
                      ox, oy, oz,
                      dx, dy, dz,
                      nw, ow, dw,
                      ntx, nty, padx, pady,
                      dtmax, nrmax,
                      slo.mutable_data(),
                      nexp,
                      nsrc.mutable_data(), srcy.mutable_data(), srcx.mutable_data(),
                      nrec.mutable_data(), recy.mutable_data(), recx.mutable_data(),
                      dat.mutable_data(), wav.mutable_data(), img.mutable_data(),
                      nthrds, verb);
      },
      py::arg("nx"), py::arg("ny"), py::arg("nz"),
      py::arg("ox"), py::arg("oy"), py::arg("oz"),
      py::arg("dx"), py::arg("dy"), py::arg("dz"),
      py::arg("nw"), py::arg("ow"), py::arg("dw"),
      py::arg("ntx"), py::arg("nty"), py::arg("px"), py::arg("py"),
      py::arg("dtmax"), py::arg("nrmax"), py::arg("slo"),
      py::arg("nexp"),
      py::arg("nsrc"), py::arg("srcy"), py::arg("srcx"),
      py::arg("nrec"), py::arg("recy"), py::arg("recx"),
      py::arg("dat"), py::arg("wav"), py::arg("img"),
      py::arg("nthrds"), py::arg("verb")
      );
  m.def("ssr3migoffshots",[](
      int nx, int ny, int nz,
      float ox, float oy, float oz,
      float dx, float dy, float dz,
      int nw, float ow, float dw,
      int ntx, int nty, int padx, int pady,
      float dtmax, int nrmax,
      py::array_t<float, py::array::c_style> slo,
      int nexp,
      py::array_t<int, py::array::c_style> nsrc,
      py::array_t<float, py::array::c_style> srcy, py::array_t<float, py::array::c_style> srcx,
      py::array_t<int, py::array::c_style> nrec,
      py::array_t<float, py::array::c_style> recy, py::array_t<float, py::array::c_style> recx,
      py::array_t<std::complex<float>, py::array::c_style> dat,
      py::array_t<std::complex<float>, py::array::c_style> wav,
      int nhy, int nhx, bool sym,
      py::array_t<float, py::array::c_style> img,
      int nthrds, int verb)
      {
        ssr3_migoffshots(nx, ny, nz,
                         ox, oy, oz,
                         dx, dy, dz,
                         nw, ow, dw,
                         ntx, nty, padx, pady,
                         dtmax, nrmax,
                         slo.mutable_data(),
                         nexp,
                         nsrc.mutable_data(), srcy.mutable_data(), srcx.mutable_data(),
                         nrec.mutable_data(), recy.mutable_data(), recx.mutable_data(),
                         dat.mutable_data(), wav.mutable_data(),
                         nhy, nhx, sym, img.mutable_data(),
                         nthrds, verb);
      },
      py::arg("nx"), py::arg("ny"), py::arg("nz"),
      py::arg("ox"), py::arg("oy"), py::arg("oz"),
      py::arg("dx"), py::arg("dy"), py::arg("dz"),
      py::arg("nw"), py::arg("ow"), py::arg("dw"),
      py::arg("ntx"), py::arg("nty"), py::arg("px"), py::arg("py"),
      py::arg("dtmax"), py::arg("nrmax"), py::arg("slo"),
      py::arg("nexp"),
      py::arg("nsrc"), py::arg("srcy"), py::arg("srcx"),
      py::arg("nrec"), py::arg("recy"), py::arg("recx"),
      py::arg("dat"), py::arg("wav"),
      py::arg("nhy"), py::arg("nhx"), py::arg("sym"), py::arg("img"),
      py::arg("nthrds"), py::arg("verb")
      );
}
