/**
 * Functions for calculating FWI gradient
 * @author: Joseph Jennings
 * @version: 2018.10.05
 */

#include "hypercube_float.h"
#include "invhelp.h"

void drslc(int recz, hypercube_float *wslc, hypercube_float *dslc);

void drtslc(int recz, hypercube_float *wslc, hypercube_float *dslc);

void drslc_coords(int recz, hypercube_float *coords, hypercube_float *wslc, hypercube_float *dslc);

void dr(int recz, hypercube_float *wfld, hypercube_float *dat);

void drt(int recz, hypercube_float *wfld, hypercube_float *dat);

void shot_interp(hypercube_float* datc, hypercube_float* datf);

void shot_interpt(hypercube_float* datc, hypercube_float* datf);

/**
 * Solves the second-order 2D acoustic wave equation with finite-differences
 * and explicit time stepping
 * Assumes all receivers are at same depth and all sources are at same depth
 * Assumes sources and receivers are regular in space (X)
 * Assumes that each shot is a point source
 * @param fin a zero-padded input source time function
 * @param v input velocity
 * @param recz receiver depth
 * @param du wavefield sampling
 * @param wfld the calculated wavefield
 * @param dat the calculated data
 */
void fwdprop_wfld(hypercube_float *fin, hypercube_float *v,int bz, int bx, float alpha, hypercube_float *psol);

void fwdprop_wfld_wavcrs(hypercube_float *wav, int srcx, int srcz, hypercube_float *v,
    int bz, int bx, float alpha, hypercube_float *psol);

void fwdprop_data(hypercube_float *fin, hypercube_float *v, int bz, int bx, float alpha, int recz, hypercube_float *dat);

void fwdprop_data_wav(hypercube_float *wav, int srcz, int srcx, hypercube_float *v, int bz, int bx, 
    float alpha, int recz, hypercube_float *dat);

void fwdprop_data_coords(hypercube_float *fin, int srcz, int srcx, hypercube_float *v, int bz, int bx,
    float alpha, int recz, hypercube_float *coords, hypercube_float *dat);

void adjprop(hypercube_float *ain, hypercube_float *v, int bz, int bx, float alpha, hypercube_float *lsol);

void d2t(hypercube_float* p ,hypercube_float *d2p);

void calc_grad(hypercube_float *d2pt, hypercube_float *lsol, hypercube_float *v, hypercube_float* grad);

float gradient(hypercube_float *src, hypercube_float *vel, hypercube_float *dat, int recz, int srcz,
    int bz, int bx, float alpha, float z1, float z2, hypercube_float *grad, int nthd, bool *write_movies, invhelp *dgn);

float indgradient(hypercube_float *src, hypercube_float *vel, hypercube_float *dat, int recz, int srcz,
    int bz, int bx, float alpha, float z1, float z2, std::vector<float> stimes, std::vector<int> xss,
    hypercube_float *grad, int nthd, bool *write_movies, invhelp *dgn);

float bldgradient(hypercube_float *src, hypercube_float *vel, hypercube_float *dat, int recz, int srcz,
    int bz, int bx, float alpha, float z1, float z2, std::vector<std::vector<float>> stimes,
    std::vector<std::vector<int>> xss, std::vector<int> nb, hypercube_float *grad, int nthd, bool *write_movies, invhelp *dgn);

void indmodel_data(hypercube_float *src, hypercube_float *vel, int recz, int srcz,
    int bz, int bx, float alpha, std::vector<float> stimes, std::vector<int> xss, int nthd, hypercube_float *idat);

void model_data(hypercube_float *src, hypercube_float *vel, int recz, int srcz,
    int bz, int bx, float alpha, int nthd, hypercube_float *dat);

void model_data_coords(hypercube_float *src, hypercube_float *vel, int recz, hypercube_float *reccoords,
    int srcz, hypercube_float *srccoords, int bz, int bx, float alpha, int nthd, hypercube_float *dat);

