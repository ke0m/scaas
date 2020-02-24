/**
 * Python interface to scaas2d.cpp
 * @author: Joseph Jennings
 * @version: 2019.11.11
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
      .def("fwdprop_oneshot",[](scaas2d &sca2d,
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
                sca2d.fwdprop_oneshot(src.mutable_data(), srcxs.mutable_data(), srczs.mutable_data(), nsrc,
                    recxs.mutable_data(), reczs.mutable_data(), nrec, vel.mutable_data(), dat.mutable_data());
              },
              py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
              py::arg("recxs"), py::arg("reczs"), py::arg("nrec"),
              py::arg("vel"), py::arg("dat")
          )
      .def("fwdprop_multishot",[](scaas2d &sca2d,
              py::array_t<float, py::array::c_style> src,
              py::array_t<int, py::array::c_style> srcxs,
              py::array_t<int, py::array::c_style> srczs,
              py::array_t<int, py::array::c_style> nsrc,
              py::array_t<int, py::array::c_style> recxs,
              py::array_t<int, py::array::c_style> reczs,
              py::array_t<int, py::array::c_style> nrec,
              int nex,
              py::array_t<float, py::array::c_style> vel,
              py::array_t<float, py::array::c_style> dat,
              int nthrds
              )
              {
                sca2d.fwdprop_multishot(src.mutable_data(),srcxs.mutable_data(), srczs.mutable_data(), nsrc.mutable_data(),
                    recxs.mutable_data(), reczs.mutable_data(), nrec.mutable_data(), nex,
                    vel.mutable_data(), dat.mutable_data(), nthrds);
              },
              py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
              py::arg("recxs"), py::arg("reczs"), py::arg("nrec"), py::arg("nex"),
              py::arg("vel"), py::arg("dat"), py::arg("nthrds")
          )
      .def("fwdprop_wfld",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> src,
               py::array_t<int, py::array::c_style> srcxs,
               py::array_t<int, py::array::c_style> srczs,
               int nsrc,
               py::array_t<float, py::array::c_style> vel,
               py::array_t<float, py::array::c_style> psol
               )
               {
                  sca2d.fwdprop_wfld(src.mutable_data(),srcxs.mutable_data(),srczs.mutable_data(),nsrc,vel.mutable_data(),
                      psol.mutable_data());
               },
               py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
               py::arg("vel"), py::arg("psol")
          )
      .def("fwdprop_lapwfld",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> src,
               py::array_t<int, py::array::c_style> srcxs,
               py::array_t<int, py::array::c_style> srczs,
               int nsrc,
               py::array_t<float, py::array::c_style> vel,
               py::array_t<float, py::array::c_style> lappsol
               )
               {
                  sca2d.fwdprop_lapwfld(src.mutable_data(),srcxs.mutable_data(),srczs.mutable_data(),nsrc,vel.mutable_data(),
                      lappsol.mutable_data());
               },
               py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
               py::arg("vel"), py::arg("lappsol")
          )
      .def("adjprop_wfld",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> asrc,
               py::array_t<int, py::array::c_style> recxs,
               py::array_t<int, py::array::c_style> reczs,
               int nrec,
               py::array_t<float, py::array::c_style> vel,
               py::array_t<float, py::array::c_style> lsol
               )
               {
                  sca2d.adjprop_wfld(asrc.mutable_data(),recxs.mutable_data(),reczs.mutable_data(),nrec,vel.mutable_data(),
                      lsol.mutable_data());
               },
               py::arg("asrc"), py::arg("recxs"), py::arg("reczs"), py::arg("nrec"),
               py::arg("vel"), py::arg("lsol")
          )
      .def("d2t",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> p,
               py::array_t<float, py::array::c_style> d2p
               )
               {
                 sca2d.d2t(p.mutable_data(),d2p.mutable_data());
               },
               py::arg("p"), py::arg("d2p")
          )
      .def("d2x",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> p,
               py::array_t<float, py::array::c_style> d2p
               )
               {
                 sca2d.d2x(p.mutable_data(),d2p.mutable_data());
               },
               py::arg("p"), py::arg("d2p")
          )
      .def("lapimg",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> img,
               py::array_t<float, py::array::c_style> lap
               )
               {
                 sca2d.lapimg(img.mutable_data(), lap.mutable_data());
               },
               py::arg("img"), py::arg("lap")
          )
      .def("calc_grad_d2t",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> d2t,
               py::array_t<float, py::array::c_style> lsol,
               py::array_t<float, py::array::c_style> v,
               py::array_t<float, py::array::c_style> grad
               )
               {
                 sca2d.calc_grad_d2t(d2t.mutable_data(),lsol.mutable_data(),v.mutable_data(),grad.mutable_data());
               },
               py::arg("d2t"),py::arg("lsol"),py::arg("v"),py::arg("grad")
          )
      .def("calc_grad_d2x",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> d2x,
               py::array_t<float, py::array::c_style> lsol,
               py::array_t<float, py::array::c_style> src,
               py::array_t<int,   py::array::c_style> srcxs,
               py::array_t<int,   py::array::c_style> srczs,
               int nsrc,
               py::array_t<float, py::array::c_style> v,
               py::array_t<float, py::array::c_style> grad
               )
               {
                 sca2d.calc_grad_d2x(d2x.mutable_data(),lsol.mutable_data(),
                     src.mutable_data(), srcxs.mutable_data(), srczs.mutable_data(), nsrc,
                     v.mutable_data(), grad.mutable_data());
               },
               py::arg("d2x"), py::arg("lsol"), py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
               py::arg("v"), py::arg("grad")
          )
      .def("shot_interp",[](scaas2d &sca2d,
               int nrec,
               py::array_t<float, py::array::c_style> datc,
               py::array_t<float, py::array::c_style> datf
               )
               {
                 sca2d.shot_interp(nrec,datc.mutable_data(),datf.mutable_data());
               },
               py::arg("nrec"), py::arg("datc"), py::arg("datf")
          )
      .def("gradient_oneshot",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> src,
               py::array_t<int, py::array::c_style> srcxs,
               py::array_t<int, py::array::c_style> srczs,
               int nsrc,
               py::array_t<float, py::array::c_style> asrc,
               py::array_t<int, py::array::c_style> recxs,
               py::array_t<int, py::array::c_style> reczs,
               int nrec,
               py::array_t<float, py::array::c_style> vel,
               py::array_t<float, py::array::c_style> grad
               )
               {
                  sca2d.gradient_oneshot(src.mutable_data(),srcxs.mutable_data(),srczs.mutable_data(),nsrc,
                      asrc.mutable_data(), recxs.mutable_data(), reczs.mutable_data(), nrec,
                      vel.mutable_data(), grad.mutable_data());
               },
               py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
               py::arg("asrc"), py::arg("recxs"), py::arg("reczs"), py::arg("nrec"),
               py::arg("vel"), py::arg("grad")
          )
      .def("gradient_multishot",[] (scaas2d &sca2d,
              py::array_t<float, py::array::c_style> src,
              py::array_t<int, py::array::c_style> srcxs,
              py::array_t<int, py::array::c_style> srczs,
              py::array_t<int, py::array::c_style> nsrcs,
              py::array_t<float, py::array::c_style> asrc,
              py::array_t<int, py::array::c_style> recxs,
              py::array_t<int, py::array::c_style> reczs,
              py::array_t<int, py::array::c_style> nrecs,
              int nex,
              py::array_t<float, py::array::c_style> vel,
              py::array_t<float, py::array::c_style> grad,
              int nthrds
              )
              {
                sca2d.gradient_multishot(src.mutable_data(), srcxs.mutable_data(), srczs.mutable_data(), nsrcs.mutable_data(),
                    asrc.mutable_data(), recxs.mutable_data(), reczs.mutable_data(), nrecs.mutable_data(), nex,
                    vel.mutable_data(), grad.mutable_data(), nthrds);
              },
              py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrcs"),
              py::arg("asrc"), py::arg("recxs"), py::arg("reczs"), py::arg("nrecs"),
              py::arg("nex"), py::arg("vel"), py::arg("grad"), py::arg("nthrds")
         )
      .def("brnfwd_oneshot",[] (scaas2d &sca2d,
              py::array_t<float, py::array::c_style> src,
              py::array_t<int, py::array::c_style> srcxs,
              py::array_t<int, py::array::c_style> srczs,
              int nsrc,
              py::array_t<int, py::array::c_style> recxs,
              py::array_t<int, py::array::c_style> reczs,
              int nrec,
              py::array_t<float, py::array::c_style> vel,
              py::array_t<float, py::array::c_style> dvel,
              py::array_t<float, py::array::c_style> ddat
             )
             {
               sca2d.brnfwd_oneshot(src.mutable_data(), srcxs.mutable_data(), srczs.mutable_data(), nsrc,
                   recxs.mutable_data(), reczs.mutable_data(), nrec, vel.mutable_data(), dvel.mutable_data(), ddat.mutable_data());
             },
             py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
             py::arg("recxs"), py::arg("reczs"), py::arg("nrec"),
             py::arg("vel"), py::arg("dvel"), py::arg("ddat")
        )
      .def("brnfwd",[](scaas2d &sca2d,
             py::array_t<float, py::array::c_style> src,
             py::array_t<int, py::array::c_style> srcxs,
             py::array_t<int, py::array::c_style> srczs,
             py::array_t<int, py::array::c_style> nsrc,
             py::array_t<int, py::array::c_style> recxs,
             py::array_t<int, py::array::c_style> reczs,
             py::array_t<int, py::array::c_style> nrec,
             int nex,
             py::array_t<float, py::array::c_style> vel,
             py::array_t<float, py::array::c_style> dvel,
             py::array_t<float, py::array::c_style> ddat,
             int nthrds
            )
            {
             sca2d.brnfwd(src.mutable_data(),srcxs.mutable_data(), srczs.mutable_data(), nsrc.mutable_data(),
                     recxs.mutable_data(), reczs.mutable_data(), nrec.mutable_data(), nex,
                     vel.mutable_data(), dvel.mutable_data(), ddat.mutable_data(), nthrds);
            },
            py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
            py::arg("recxs"), py::arg("reczs"), py::arg("nrec"), py::arg("nex"),
            py::arg("vel"), py::arg("dvel"), py::arg("ddat"), py::arg("nthrds")
          )
      .def("brnadj_oneshot",[](scaas2d &sca2d,
             py::array_t<float, py::array::c_style> src,
             py::array_t<int, py::array::c_style> srcxs,
             py::array_t<int, py::array::c_style> srczs,
             int nsrc,
             py::array_t<int, py::array::c_style> recxs,
             py::array_t<int, py::array::c_style> reczs,
             int nrec,
             py::array_t<float, py::array::c_style> vel,
             py::array_t<float, py::array::c_style> dv,
             py::array_t<float, py::array::c_style> ddat
             )
             {
                sca2d.brnadj_oneshot(src.mutable_data(),srcxs.mutable_data(),srczs.mutable_data(),nsrc,
                    recxs.mutable_data(), reczs.mutable_data(), nrec,
                    vel.mutable_data(), dv.mutable_data(), ddat.mutable_data());
             },
             py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
             py::arg("recxs"), py::arg("reczs"), py::arg("nrec"),
             py::arg("vel"), py::arg("dv"), py::arg("ddat")
        )
      .def("brnadj",[] (scaas2d &sca2d,
             py::array_t<float, py::array::c_style> src,
             py::array_t<int, py::array::c_style> srcxs,
             py::array_t<int, py::array::c_style> srczs,
             py::array_t<int, py::array::c_style> nsrcs,
             py::array_t<int, py::array::c_style> recxs,
             py::array_t<int, py::array::c_style> reczs,
             py::array_t<int, py::array::c_style> nrecs,
             int nex,
             py::array_t<float, py::array::c_style> vel,
             py::array_t<float, py::array::c_style> dvel,
             py::array_t<float, py::array::c_style> ddat,
             int nthrds
             )
             {
               sca2d.brnadj(src.mutable_data(), srcxs.mutable_data(), srczs.mutable_data(), nsrcs.mutable_data(),
                   recxs.mutable_data(), reczs.mutable_data(), nrecs.mutable_data(), nex,
                    vel.mutable_data(), dvel.mutable_data(), ddat.mutable_data(), nthrds);
             },
             py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrcs"),
             py::arg("recxs"), py::arg("reczs"), py::arg("nrecs"), py::arg("nex"),
             py::arg("vel"), py::arg("dvel"), py::arg("ddat"), py::arg("nthrds")
          )
      .def("brnoffadj_oneshot",[](scaas2d &sca2d,
               py::array_t<float, py::array::c_style> src,
               py::array_t<int, py::array::c_style> srcxs,
               py::array_t<int, py::array::c_style> srczs,
               int nsrc,
               py::array_t<int, py::array::c_style> recxs,
               py::array_t<int, py::array::c_style> reczs,
               int nrec,
               py::array_t<float, py::array::c_style> vel,
               int nrh,
               py::array_t<float, py::array::c_style> dv,
               py::array_t<float, py::array::c_style> ddat
               )
               {
                  sca2d.brnoffadj_oneshot(src.mutable_data(),srcxs.mutable_data(),srczs.mutable_data(),nsrc,
                      recxs.mutable_data(), reczs.mutable_data(), nrec,
                      vel.mutable_data(), nrh, dv.mutable_data(), ddat.mutable_data());
               },
               py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrc"),
               py::arg("recxs"), py::arg("reczs"), py::arg("nrec"),
               py::arg("vel"), py::arg("nrh"), py::arg("dv"), py::arg("ddat")
          )
      .def("brnoffadj",[] (scaas2d &sca2d,
               py::array_t<float, py::array::c_style> src,
               py::array_t<int, py::array::c_style> srcxs,
               py::array_t<int, py::array::c_style> srczs,
               py::array_t<int, py::array::c_style> nsrcs,
               py::array_t<int, py::array::c_style> recxs,
               py::array_t<int, py::array::c_style> reczs,
               py::array_t<int, py::array::c_style> nrecs,
               int nex,
               py::array_t<float, py::array::c_style> vel,
               int rnh,
               py::array_t<float, py::array::c_style> dvel,
               py::array_t<float, py::array::c_style> ddat,
               int nthrds
               )
               {
                 sca2d.brnoffadj(src.mutable_data(), srcxs.mutable_data(), srczs.mutable_data(), nsrcs.mutable_data(),
                     recxs.mutable_data(), reczs.mutable_data(), nrecs.mutable_data(), nex,
                      vel.mutable_data(), rnh, dvel.mutable_data(), ddat.mutable_data(), nthrds);
               },
               py::arg("src"), py::arg("srcxs"), py::arg("srczs"), py::arg("nsrcs"),
               py::arg("recxs"), py::arg("reczs"), py::arg("nrecs"), py::arg("nex"),
               py::arg("vel"), py::arg("rnh"), py::arg("dvel"), py::arg("ddat"), py::arg("nthrds")
          );
}
