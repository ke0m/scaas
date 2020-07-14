/**
 * Python interface to slowness transform
 * @author: Joseph Jennings
 * @version: 2020.07.13
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "slow.h"

namespace py = pybind11;

PYBIND11_MODULE(slowtfm,m) {
  m.doc() = "Applies a mute to seismic gathers";
  m.def("slowtfmfwd",[](
      int nq, float oq, float dq, int nz, float oz, float dz,
      int nx, float ox, float dx, int nt, float ot, float dt,
      py::array_t<float, py::array::c_style> mod,
      py::array_t<float, py::array::c_style> dat
      )
      {
        slowforward(nq,oq,dq,nz,oz,dz,
                    nx,ox,dx,nt,ot,dt,
                    mod.mutable_data(),dat.mutable_data());
      },
      py::arg("nq"), py::arg("oq"), py::arg("dq"),
      py::arg("nz"), py::arg("oz"), py::arg("dz"),
      py::arg("nx"), py::arg("ox"), py::arg("dx"),
      py::arg("nt"), py::arg("ot"), py::arg("dt"),
      py::arg("mod"), py::arg("dat")
      );
  m.def("slowtfmadj",[](
        int nq, float oq, float dq, int nz, float oz, float dz,
        int nx, float ox, float dx, int nt, float ot, float dt,
        py::array_t<float, py::array::c_style> mod,
        py::array_t<float, py::array::c_style> dat
        )
        {
          slowadjoint(nq,oq,dq,nz,oz,dz,
                      nx,ox,dx,nt,ot,dt,
                      mod.mutable_data(),dat.mutable_data());
        },
        py::arg("nq"), py::arg("oq"), py::arg("dq"),
        py::arg("nz"), py::arg("oz"), py::arg("dz"),
        py::arg("nx"), py::arg("ox"), py::arg("dx"),
        py::arg("nt"), py::arg("ot"), py::arg("dt"),
        py::arg("mod"), py::arg("dat")
        );
}
