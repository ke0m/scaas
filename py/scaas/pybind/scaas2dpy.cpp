/**
 * Python interface to scaas2d.cpp
 * @author: Joseph Jennings
 * @version: 2019.11.01
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "scaas2d.h"

namespace py = pybind11;

PYBIND11_MODULE(scaas2dpy,m) {
  m.doc() = "2D scalar acoustic wave propagation";

  py::class_<scaas2d>(m,"scaas2d")
      .def(py::init<int,int,int,float,float,float,float,int,int,float>(),
          py::arg("nt"),py::arg("nx"),py::arg("nz"),
          py::arg("dt"),py::arg("dx"),py::arg("dz"),py::arg("dtu")=float(0.001),
          py::arg("bx")=int(50),py::arg("bz")=int(50),py::arg("alpha")=float(0.99))
      .def("get_info", &scaas2d::get_info)
      .def("fwdprop_data",[](scaas2d &sca2d,
              py::array_t<float, py::array::c_style> src,
              py::array_t<int, py::array::c_style> srcxs,
              py::array_t<int, py::array::c_style> srczs,
              int nsrc,
              py::array_t<int, py::array::c_style> recxs,
              py::array_t<int, py::array::c_style> reczs,
              int nrec,
              py::array_t<float, py::array::c_style> vel,
              py::array_t<float, py::array::c_style> dat
            )
              {
                sca2d.fwdprop_data(src.mutable_data(), srcxs.mutable_data(), srczs.mutable_data(), nsrc,
                    recxs.mutable_data(), reczs.mutable_data(), nrec, vel.mutable_data(), dat.mutable_data());
              },
              py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
              py::arg("recxs"), py::arg("reczs"), py::arg("nrec"),
              py::arg("vel"), py::arg("dat")
          );
}
