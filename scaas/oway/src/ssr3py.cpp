/**
 * Python interface to ssr3.cpp
 * @author: Joseph Jennings
 * @version: 2020.11.04
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
                    float,int,int>(),
          py::arg("nx"),py::arg("ny"),py::arg("nz"),
          py::arg("dx"),py::arg("dy"),py::arg("dz"),
          py::arg("nw"), py::arg("ow"), py::arg("dw"), py::arg("eps"),
          py::arg("ntx"),py::arg("nty"),py::arg("px"),py::arg("py"),
          py::arg("dtmax"),py::arg("nrmax"),py::arg("nthrds"))
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
              py::array_t<std::complex<float>, py::array::c_style> dat,
              bool verb
              )
              {
                sr3d.ssr3ssf_modallw(ref.mutable_data(), wav.mutable_data(), dat.mutable_data(), verb);
              },
              py::arg("ref"), py::arg("wav"), py::arg("dat"), py::arg("verb")
          )
      .def("modonew",[](ssr3 &sr3d,
             int iw,
             py::array_t<float, py::array::c_style> ref,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             int ithrd
             )
             {
               sr3d.ssr3ssf_modonew(iw, ref.mutable_data(), wav.mutable_data(), dat.mutable_data(), ithrd);
             },
             py::arg("iw"), py::arg("ref"), py::arg("wav"), py::arg("dat"), py::arg("ithrd")
          )
      .def("restrict_data",[](ssr3 &sr3d,
             int nrec,
             py::array_t<float, py::array::c_style> recy,
             py::array_t<float, py::array::c_style> recx,
             float oy,
             float ox,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> rec
             )
             {
               sr3d.restrict_data(nrec, recy.mutable_data(), recx.mutable_data(), oy, ox,
                                  dat.mutable_data(), rec.mutable_data());
             },
             py::arg("nrec"), py::arg("recy"), py::arg("recx"), py::arg("oy"), py::arg("ox"),
             py::arg("dat"), py::arg("rec")
          )
      .def("inject_data",[](ssr3 &sr3d,
             int nrec,
             py::array_t<float, py::array::c_style> recy,
             py::array_t<float, py::array::c_style> recx,
             float oy,
             float ox,
             py::array_t<std::complex<float>, py::array::c_style> rec,
             py::array_t<std::complex<float>, py::array::c_style> dat
             )
             {
               sr3d.inject_data(nrec, recy.mutable_data(), recx.mutable_data(), oy, ox,
                                rec.mutable_data(), dat.mutable_data());
             },
             py::arg("nrec"), py::arg("recx"), py::arg("recy"), py::arg("oy"), py::arg("ox"),
             py::arg("rec"), py::arg("dat")
          )
      .def("migallw",[](ssr3 &sr3d,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             py::array_t<float, py::array::c_style> img,
             bool verb
             )
             {
               sr3d.ssr3ssf_migallw(dat.mutable_data(), wav.mutable_data(), img.mutable_data(), verb);
             },
             py::arg("dat"), py::arg("wav"), py::arg("img"), py::arg("verb")
          )
      .def("migonew",[](ssr3 &sr3d,
             int iw,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             py::array_t<float, py::array::c_style> img,
             int ithrd
             )
             {
               sr3d.ssr3ssf_migonew(iw, dat.mutable_data(), wav.mutable_data(), img.mutable_data(), ithrd);
             },
             py::arg("iw"), py::arg("dat"), py::arg("wav"), py::arg("img"), py::arg("ithrd")
          )
      .def("migoffallw",[](ssr3 &sr3d,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             py::array_t<float, py::array::c_style> img,
             bool verb
             )
             {
               sr3d.ssr3ssf_migoffallw(dat.mutable_data(), wav.mutable_data(), img.mutable_data(),verb);
             },
             py::arg("dat"), py::arg("wav"), py::arg("img"),py::arg("verb")
          )
      .def("migoffonew",[](ssr3 &sr3d,
             int iw,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             int bly,
             int ely,
             int blx,
             int elx,
             py::array_t<float, py::array::c_style> img,
             int ithrd
             )
             {
               sr3d.ssr3ssf_migoffonew(iw, dat.mutable_data(), wav.mutable_data(),
                                       bly, ely, blx, elx, img.mutable_data(), ithrd);
             },
             py::arg("iw"), py::arg("dat"), py::arg("wav"), py::arg("bly"), py::arg("ely"),
             py::arg("blx"), py::arg("elx"), py::arg("img"), py::arg("ithrd")
          )
      .def("migoffallwbig",[](ssr3 &sr3d,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             py::array_t<float, py::array::c_style> img,
             bool verb
             )
             {
               sr3d.ssr3ssf_migoffallwbig(dat.mutable_data(), wav.mutable_data(), img.mutable_data(),verb);
             },
             py::arg("dat"), py::arg("wav"), py::arg("img"),py::arg("verb")
          )
      .def("migoffonewbig",[](ssr3 &sr3d,
             int iw,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wav,
             int bly,
             int ely,
             int blx,
             int elx,
             py::array_t<float, py::array::c_style> img,
             int ithrd
             )
             {
               sr3d.ssr3ssf_migoffonewbig(iw, dat.mutable_data(), wav.mutable_data(),
                                          bly, ely, blx, elx, img.mutable_data(), ithrd);
             },
             py::arg("iw"), py::arg("dat"), py::arg("wav"), py::arg("bly"), py::arg("ely"),
             py::arg("blx"), py::arg("elx"), py::arg("img"), py::arg("ithrd")
          )
       .def("set_ext",[](ssr3 &sr3d,
             int nhy,
             int nhx,
             bool sym,
						 bool alloc
             )
             {
               sr3d.set_ext(nhy, nhx, sym, alloc);
             },
             py::arg("nhy"), py::arg("nhx"), py::arg("sym"), py::arg("alloc")
          )
      .def("del_ext",[](ssr3 &sr3d)
          {
             sr3d.del_ext();
          }
          )
      .def("modallwzo",[](ssr3 &sr3d,
             py::array_t<float, py::array::c_style> img,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             bool verb
             )
             {
               sr3d.ssr3ssf_modallwzo(img.mutable_data(), dat.mutable_data(), verb);
             },
             py::arg("img"), py::arg("dat"), py::arg("verb")
         )
     .def("modonewzo",[](ssr3 &sr3d,
             int iw,
             py::array_t<float, py::array::c_style> img,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             int ithrd
             )
             {
               sr3d.ssr3ssf_modonewzo(iw, img.mutable_data(), dat.mutable_data(), ithrd);
             },
             py::arg("iw"), py::arg("img"), py::arg("dat"), py::arg("ithrd")
         )
     .def("migallwzo",[](ssr3 &sr3d,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<float, py::array::c_style> img,
             bool verb
             )
             {
               sr3d.ssr3ssf_migallwzo(dat.mutable_data(), img.mutable_data(), verb);
             },
             py::arg("dat"), py::arg("img"), py::arg("verb")
         )
     .def("migonewzo",[](ssr3 &sr3d,
             int iw,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<float, py::array::c_style> img,
             int ithrd
             )
             {
                sr3d.ssr3ssf_migonewzo(iw, dat.mutable_data(), img.mutable_data(), ithrd);
             },
             py::arg("iw"), py::arg("dat"), py::arg("img"), py::arg("ithrd")
         )
     .def("fwfallwzo",[](ssr3 &sr3d,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wfl,
             bool verb
             )
             {
                sr3d.ssr3ssf_fwfallwzo(dat.mutable_data(), wfl.mutable_data(), verb);
             },
             py::arg("dat"), py::arg("wfl"), py::arg("verb")
         )
     .def("fwfonewzo",[](ssr3 &sr3d,
             int iw,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wfl,
             int ithrd
             )
             {
                sr3d.ssr3ssf_fwfonewzo(iw, dat.mutable_data(), wfl.mutable_data(), ithrd);
             },
             py::arg("iw"), py::arg("dat"), py::arg("wfl"), py::arg("ithrd")
         )
     .def("awfallwzo",[](ssr3 &sr3d,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wfl,
             bool verb
             )
             {
                sr3d.ssr3ssf_awfallwzo(dat.mutable_data(), wfl.mutable_data(), verb);
             },
             py::arg("dat"), py::arg("wfl"), py::arg("verb")
         )
     .def("awfonewzo",[](ssr3 &sr3d,
             int iw,
             py::array_t<std::complex<float>, py::array::c_style> dat,
             py::array_t<std::complex<float>, py::array::c_style> wfl,
             int ithrd
             )
             {
                sr3d.ssr3ssf_awfonewzo(iw, dat.mutable_data(), wfl.mutable_data(), ithrd);
             },
             py::arg("iw"), py::arg("dat"), py::arg("wfl"), py::arg("ithd")
         )
     .def("fwemvaallw", [] (ssr3 &sr3d,
              py::array_t<std::complex<float>, py::array::c_style> src,
              py::array_t<std::complex<float>, py::array::c_style> rec,
              py::array_t<std::complex<float>, py::array::c_style> dslo,
              py::array_t<std::complex<float>, py::array::c_style> dimg,
              bool verb
              )
              {
                sr3d.ssr3ssf_fwemvaallw(src.mutable_data(), rec.mutable_data(),
                                        dslo.mutable_data(), dimg.mutable_data(), verb);
              },
              py::arg("src"), py::arg("rec"), py::arg("dslo"), py::arg("dimg"), py::arg("verb")
         )
     .def("awemvaallw", [] (ssr3 &sr3d,
              py::array_t<std::complex<float>, py::array::c_style> src,
              py::array_t<std::complex<float>, py::array::c_style> rec,
              py::array_t<std::complex<float>, py::array::c_style> dslo,
              py::array_t<std::complex<float>, py::array::c_style> dimg,
              bool verb
              )
              {
                sr3d.ssr3ssf_awemvaallw(src.mutable_data(), rec.mutable_data(),
                                        dslo.mutable_data(), dimg.mutable_data(), verb);
              },
              py::arg("src"), py::arg("rec"), py::arg("dslo"), py::arg("dimg"), py::arg("verb")
         );
      m.def("interp_slow",[] (int nz,
              int nvy, float ovy, float dvy,
              int nvx, float ovx, float dvx,
              int ny , float oy , float dy ,
              int nx , float ox , float dx ,
              py::array_t<float, py::array::c_style> sloin,
              py::array_t<float, py::array::c_style> sloot
              )
              {
                interp_slow(nz,
                            nvy, ovy, dvy, nvx, ovx, dvx,
                            ny,  oy,  dy,  nx,  ox,  dx,
                            sloin.mutable_data(), sloot.mutable_data());
              },
              py::arg("nz"),
              py::arg("nvy"), py::arg("ovy"), py::arg("dvy"),
              py::arg("nvx"), py::arg("ovx"), py::arg("dvx"),
              py::arg("ny") , py::arg("oy") , py::arg("dy") ,
              py::arg("nx") , py::arg("ox") , py::arg("dx") ,
              py::arg("sloin"), py::arg("sloot")
          );
}
