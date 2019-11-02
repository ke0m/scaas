#include <stdio.h>
#include <math.h>
#include <cstring>
#include "scaas2d.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

scaas2d::scaas2d(int nt, int nx, int nz, float dt, float dx, float dz, float dtu, int bx, int bz, float alpha) {
  /* Lengths */
  _nt = nt; _nx = nx; _nz = nz; _onestp = _nx*_nz;
  /* Samplings */
  _dt = dt; _dx = dx; _dz = dz; _dtu = dtu;
  _idx2 = 1/(_dx*_dx); _idz2 = 1/(_dz*_dz);
  /* Taper parameters */
  _bx = bx; _bz = bz; _alpha = alpha;
  /* Propagation vs data length */
  _skip = static_cast<int>(_dt/_dtu); _ntu = _nt*_skip;
  printf("skip=%d\n",_skip);
}

void scaas2d::drslc(int *recxs, int *reczs, int nrec, float *wslc, float *dslc) {

  /* Loop over all receivers for the time slice */
  //plt::imshow((const float*)wslc,_nx,_nz,1); plt::show();
  //printf("nz=%d nx=%d\n",_nz,_nx);
  for(int ir = 0; ir < nrec; ++ir) {
    /* Grab at the receiver gridpoints (already interpolated) */
    //printf("recz=%d recx=%d\n",reczs[ir],recxs[ir]);
    dslc[ir] = wslc[recxs[ir]*_nz + reczs[ir]];
  }
}

void scaas2d::fwdprop_data(float *src, int *srcxs, int *srczs, int nsrc, int *recxs, int *reczs, int nrec, float *vel, float *dat) {

  //std::vector<float>v {src,src+_ntu};
  //plt::plot(v); plt::show();
  //plt::imshow((const float*)vel,_nx,_nz,1); plt::show();

//  for(int irec = 0; irec < 2*nrec; ++irec) {
//    dat[irec] = 1;
//  }
//  plt::imshow((const float*)dat,_nt,nrec,1); plt::show();

  /* Precompute velocity dt^2 coefficient */
  float *v2dt2 = new float[_onestp]();
  //TODO: This should be done outside
  for(int k = 0; k < _onestp; ++k) { v2dt2[k] = vel[k]*vel[k]*_dtu*_dtu; };

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  //TODO: This should be done outside
  float *tap = new float[_onestp]();
  build_taper(tap);

  /* Allocate memory for wavefield slices */
  //TODO: This could be done outside
  float *pre  = new float[_onestp]();
  float *cur  = new float[_onestp]();
  float *sou1 = new float[_onestp]();
  float *sou2 = new float[_onestp]();
  float *dslc = new float[_nt*nrec]();
  float *tmp;

  /* Initial conditions */
  memcpy(dat,dslc,sizeof(float)*nrec); //p(x,0) = 0.0

  /* Inject sources */
  for(int isrc = 0; isrc < nsrc; ++isrc) {
    printf("srcz=%d srcx=%d\n",srczs[isrc],srcxs[isrc]);
    sou1[srcxs[isrc]*_nz + srczs[isrc]] = src[isrc*_ntu + 0]/(_dx*_dz);
  }
  for(int k = 0; k < _onestp; ++k) { sou1[k] *= v2dt2[k]; };
  apply_taper(tap,sou1,pre);
  drslc(recxs, reczs, nrec, sou1, dslc);
  memcpy(&dat[nrec],dslc,sizeof(float)*nrec); // p(x,1) = v^2*dt^2*f(0)
  memcpy(cur,sou1,sizeof(float)*_onestp);

  //printf("nx=%d nz=%d\n",_nx,_nz);
  int kt = 1;
  for(int it = 2; it < _ntu; ++it) {
    /* Calculate source term */
    for(int isrc = 0; isrc < nsrc; ++isrc) {
      sou1[srcxs[isrc]*_nz + srczs[isrc]] = src[isrc*_ntu + it]/(_dx*_dz);
    }
    /* Apply laplacian */
    //laplacian10(_nz,_nx,_idx2,_idz2,cur,sou2);
    laplacianFWDISPC(_nx,_nz,_idx2,_idz2,cur,sou2);
    //if(it < 10) {
    //  plt::imshow((const float*)sou2,_nx,_nz,1); plt::show();
    //}
    /* Advance wavefields */
    for(int k = 0; k < _onestp; ++k) {
      pre[k] = v2dt2[k]*sou1[k] + 2*cur[k] + v2dt2[k]*sou2[k] - pre[k];
    }
//    if(it%100 == 0) {
//      printf("it=%d\n",it);
//      plt::subplot(1,2,1);
//      plt::imshow((const float*)vel,_nx,_nz,1);
//      plt::subplot(1,2,2);
//      plt::imshow((const float*)pre,_nx,_nz,1); plt::show();
//    }
    /* Apply taper */
    apply_taper(tap,pre,cur);
    /* Save data on coarse time grid */
    if(it%_skip == 0) {
      /* Apply delta function operator */
      drslc(recxs,reczs,nrec,pre,dslc);
      /* Copy to data vector */
      memcpy(&dat[kt*nrec],dslc,sizeof(float)*nrec);
      kt++;
    }
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Free memory */
  delete[] v2dt2;
  delete[] pre; delete[] cur; delete[] sou1;
  delete[] sou2; delete[] dslc; delete[] tap;

}

void scaas2d::build_taper(float *tap) {
  /* Fill the the taper with ones */
  for(int k = 0; k < _onestp; ++k) { tap[k] = 1.0; };

  /* Always use 10th order */
  int ord = 10;
  int shift = ord/2;

  /* Left and Right */
  for(int ix = shift; ix < _bx+shift; ++ix) {
    for(int iz = shift; iz < _nz-shift; ++iz) {
      double tapval = 1.0 * (_bx - (ix - shift))/(_bx);
      tapval = _alpha + (1.0 - _alpha) * cos(M_PI*tapval);
      tap[ix        *_nz + iz] = tapval;
      tap[(_nx-ix-1)*_nz + iz] = tapval;
    }
  }

  /* Top and Bottom */
  for(int iz = shift; iz < _bz+shift; ++iz) {
    for(int ix = shift; ix < _nx-shift; ++ix) {
      double tapval = 1.0 * (_bz - (iz - shift))/(_bz);
      tapval = _alpha + (1.0 - _alpha) * cos(M_PI*tapval);
      tap[ix*_nz + iz      ] *= tapval;
      tap[ix*_nz + _nz-iz-1] *= tapval;
    }
  }
}

void scaas2d::apply_taper(float *tap, float *cur, float *nex) {

  /* Always use 10th order */
  int ord = 10;
  int shift = ord/2;
  int nshift = (_nx-_bx-shift)*_nz;

  /* Left and Right application */
  for(int ileft = shift; ileft < (_bx+shift)*(_nz); ++ileft) {
    int irite = nshift + ileft;
    cur[ileft] *= tap[ileft];
    nex[ileft] *= tap[ileft];
    cur[irite] *= tap[irite];
    nex[irite] *= tap[irite];
  }

  /* Top and Bottom application */
  for(int ix = _bx+shift; ix < _nx-_bx-shift; ++ix) {
    for(int iz = shift; iz < _bz+shift; ++iz) {
      cur[ix*_nz +       iz] *= tap[ix*_nz +       iz];
      nex[ix*_nz +       iz] *= tap[ix*_nz +       iz];
      cur[ix*_nz + _nz-iz-1] *= tap[ix*_nz + _nz-iz-1];
      nex[ix*_nz + _nz-iz-1] *= tap[ix*_nz + _nz-iz-1];
    }
  }

}

void scaas2d::get_info() {
  printf("\n Printing Propagation parameters\n");
  printf("nt=%d, dt=%f ot=0.0\n",_nt,_dt);
  printf("nz=%d, dz=%f oz=0.0\n",_nz,_dz);
  printf("nx=%d, dx=%f ox=0.0\n",_nx,_dx);
}
