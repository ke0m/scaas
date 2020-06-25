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
      .def(py::init<int,int,int,int,int,float,float,float,float,float,float,float,float,
                    int,int,int,int,int>(),
          py::arg("nx"),py::arg("ny"),py::arg("nz"),py::arg("nh"),py::arg("nw"),
          py::arg("dx"),py::arg("dy"),py::arg("dz"),py::arg("dh"),py::arg("dw"),py::arg("dtmax"),
          py::arg("ow"),py::arg("eps"),py::arg("ntx"),py::arg("nty"),py::arg("px"),py::arg("py"),
          py::arg("nrmax"))
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
             );
}
