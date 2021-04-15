/**
 * Python interface for simple omp function
 * @author: Joseph Jennings
 * @version: 2020.07.14
 */

#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void myompfunc(int n, float scale, float *in, float *ot, int nthreads) {

  omp_set_num_threads(nthreads);
#pragma omp parallel for default(shared)
  for(int i = 0; i < n; ++i) {
    ot[i] = in[i]*scale;
  }
}

PYBIND11_MODULE(ompfunc,m) {
  m.doc() = "Dummy OpenMP function";
  m.def("ompfunc",[](
      int n,
      float scale,
      py::array_t<float, py::array::c_style> in,
      py::array_t<float, py::array::c_style> ot,
      int nthreads
      )
      {
        myompfunc(n,scale,in.mutable_data(),ot.mutable_data(),nthreads);
      },
      py::arg("n"), py::arg("scale"), py::arg("in"), py::arg("ot"), py::arg("nthreads")
      );
}
