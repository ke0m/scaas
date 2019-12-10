/**
 * 2D Scalar acoustic wave equation forward 
 * and adjoint (No SEPlib. Yay!)
 * @author: Joseph Jennings
 * @version: 2019.11.03
 **/

#ifndef SCAAS2D_H
#define SCAAS2D_H
extern "C" {
#include "laplacian10.h"
}

class scaas2d {
  public:
    /// Constructor
    scaas2d(int nt, int nx, int nz, float dt, float dx, float dz, float dtu=0.001, int bx=50, int bz=50, float alpha=0.99);
    /// Forward and adjoint wave propagation
    void fwdprop_oneshot(float *src, int *srcxs, int *srczs, int nsrc, int *recxs, int *reczs, int nrec, float *vel, float *dat);
    void fwdprop_multishot(float *src, int *srcxs, int *srczs, int *nsrc, int *recxs, int *reczs, int *nrec, int nex, float *vel, float *dat, int nthrds);
    void fwdprop_wfld(float *src, int *srcxs, int *srcsz, int nsrc, float *vel, float *psol);
    void fwdprop_lapwfld(float *src, int *srcxs, int *srcsz, int nsrc, float *vel, float *lappsol);
    void adjprop_wfld(float *asrc, int *recxs, int *reczs, int nrec, float *vel, float *lsol);
    /// Gradient helper functions
    void d2t(float* p ,float *d2tp);
    void d2x(float *p, float *d2xp);
    void calc_grad_d2t(float *d2pt, float *lsol, float *v, float * grad);
    void calc_grad_d2x(float *d2px, float *lsol, float *src, int *srcxs, int *srczs, int nsrc, float *v, float *grad);
    /// Gradient functions
    void gradient_oneshot(float *src, int *srcxs, int *srczs, int nsrc, float *asrc, int *recxs, int *reczs, int nrec, float *vel, float *grad);
    void gradient_multishot(float *src, int *srcxs, int *srczs, int *nsrcs, float *asrc, int *recxs, int *reczs, int *nrecs, int nex, float *vel, float *grad, int nthrds);
    /// Born functions
    void brnfwd_oneshot(float *src, int *srcxs, int *srczs, int nsrc, int *recxs, int *reczs, int nrec, float *vel, float *dvel, float *ddat);
    void brnfwd(float *src, int *srcxs, int *srczs, int *nsrc, int *recxs, int *reczs, int *nrec, int nex, float *vel, float *dat, int nthrds);
    void brnadj_oneshot(float *src, int *srcxs, int *srczs, int nsrc, int *recxs, int *reczs, int nrec, float *vel, float *dvel, float *ddat);
    void brnoffadj_oneshot(float *src, int *srcxs, int *srczs, int nsrc, int *recxs, int *reczs, int nrec, float *vel, int rnh, float *dvel, float *ddat);
    void brnadj(float *src, int *srcxs, int *srczs, int *nsrcs, float *asrc, int *recxs, int *reczs, int *nrecs, int nex, float *vel, float *grad, int nthrds);
    /// Shot and receiver (data) functions
    void dr(int *recx, int *recz, int nrec, float *wfld, float *dat);
    void drt(int *recx, int *recz, int nrec, float *wfld, float *dat);
    void drslc(int *recx, int *recz, int nrec, float *wslc, float *dslc);
    void drslct(int *recx, int *recz, int nrec, float *wslc, float *dslc);
    void shot_interp(int nrec, float *datc, float *datf);
    /// Miscellaneous
    void get_info();

  private:
    /// Absorbing boundary functions
    void build_taper(float *tap);
    void apply_taper(float *tap, float *cur, float *nex);
    int _nx, _nz, _nt, _ntu, _bx, _bz, _onestp, _skip;
    float _dx, _dz, _dt, _dtu, _alpha;
    float _idx2, _idz2;
};

#endif
