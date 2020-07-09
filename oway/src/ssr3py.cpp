/**
 * Python interface to ssr3.cpp
 * @author: Joseph Jennings
 * @version: 2020.06.23
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "ssr3.h"

namespace py = pybind11;

PYBIND11_MODULE(ssr3,m) {
  m.doc() = "3D one-way wave equation modeling and migration";

  py::class_<ssr3>(m,"ssr3")
      .def(py::init<int,  int,  int,
                    float,float,float,
                    int,float,float,float,
                    int,int,int,int,
                    float,int>(),
          py::arg("nx"),py::arg("ny"),py::arg("nz"),
          py::arg("dx"),py::arg("dy"),py::arg("dz"),
          py::arg("nw"), py::arg("ow"), py::arg("dw"), py::arg("eps"),
          py::arg("ntx"),py::arg("nty"),py::arg("px"),py::arg("py"),
          py::arg("dtmax"),py::arg("nrmax"))
      .def("set_slows", [] (ssr3 &sr3d,
              py::array_t<float, py::array::c_style> slo
              )
              {
                sr3d.set_slows(slo.mutable_data());
              },
              py::arg("slo")
          )
      .def("modallw",[](ssr3 &sr3d,
              py::array_t<float, py::array::c_style> ref,
              py::array_t<std::complex<float>, py::array::c_style> wav,
              py::array_t<std::complex<float>, py::array::c_style> dat
              )
              {
                sr3d.ssr3ssf_modallw(ref.mutable_data(), wav.mutable_data(), dat.mutable_data());
              },
              py::arg("ref"), py::arg("wav"), py::arg("dat")
          )
      .def("modonew",[](ssr3 &sr3d,
             int iw,
             py::array_t<float, py::array::c_style> ref,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             py::array_t<std::complex<float>, py::array::c_style> dat
             )
             {
               sr3d.ssr3ssf_modonew(iw, ref.mutable_data(), wav.mutable_data(), dat.mutable_data());
             },
             py::arg("iw"), py::arg("ref"), py::arg("wav"), py::arg("dat")
          )
      .def("migallw",[](ssr3 &sr3d,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             py::array_t<float, py::array::c_style> img
             )
             {
               sr3d.ssr3ssf_migallw(dat.mutable_data(), wav.mutable_data(), img.mutable_data());
             },
             py::arg("dat"), py::arg("wav"), py::arg("img")
          )
      .def("migonew",[](ssr3 &sr3d,
             int iw,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             py::array_t<float, py::array::c_style> img
             )
             {
               sr3d.ssr3ssf_migonew(iw, dat.mutable_data(), wav.mutable_data(), img.mutable_data());
             },
             py::arg("iw"), py::arg("dat"), py::arg("wav"), py::arg("img")
          )
      .def("migoffallw",[](ssr3 &sr3d,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             int nhx,
             int nhy,
             bool sym,
             py::array_t<float, py::array::c_style> img
             )
             {
               sr3d.ssr3ssf_migoffallw(dat.mutable_data(), wav.mutable_data(), nhx, nhy, sym, img.mutable_data());
             },
             py::arg("dat"), py::arg("wav"), py::arg("nhx"), py::arg("nhy"), py::arg("sym"), py::arg("img")
          )
      .def("migoffonew",[](ssr3 &sr3d,
             int iw,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             int bly,
             int ely,
             int blx,
             int elx,
             py::array_t<float, py::array::c_style> img
             )
             {
               sr3d.ssr3ssf_migoffonew(iw, dat.mutable_data(), wav.mutable_data(),
                                       bly, ely, blx, elx, img.mutable_data());
             },
             py::arg("iw"), py::arg("dat"), py::arg("wav"), py::arg("bly"), py::arg("ely"),
             py::arg("blx"), py::arg("elx"), py::arg("img")
          );
}
