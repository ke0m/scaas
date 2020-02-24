#include <stdio.h>
#include <math.h>
#include <cstring>
#include <omp.h>
#include "scaas2d.h"
#include "/opt/matplotlib-cpp/matplotlibcpp.h"
#include <iostream>

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
}


void scaas2d::drslc(int *recxs, int *reczs, int nrec, float *wslc, float *dslc) {

  /* Loop over all receivers for the time slice */
  for(int ir = 0; ir < nrec; ++ir) {
    /* Grab at the receiver gridpoints (already interpolated) */
    dslc[ir] = wslc[reczs[ir]*_nx + recxs[ir]];
  }
}

void scaas2d::shot_interp(int nrec, float *datc, float *datf) {

  /* Allocate memory */
  float *slc1 = new float[nrec*_nt]();
  float *slc2 = new float[nrec*_nt]();
  float *intp = new float[nrec*_nt]();

  /* Calculate coefficients */
  std::vector<float> ls1, ls2;
  for(int is = 0; is < _skip; ++is) {
    ls1.push_back(static_cast<float>(_skip-is)/static_cast<float>(_skip));
    ls2.push_back(1.0-ls1[is]);
    //ls1[is] /= _skip; ls2[is] /= _skip;
  }

  /* Interpolate from coarse to fine grid */
  int k = 0;
  for(int it = 0; it < _nt-1; ++it) {
    /* Grab values on coarse grid */
    memcpy(slc1,&datc[(it+0)*nrec],sizeof(float)*nrec);
    memcpy(slc2,&datc[(it+1)*nrec],sizeof(float)*nrec);
    for(int is = 0; is < _skip; ++is) {
      /* Interpolate onto fine grid */
      for(int ir = 0; ir < nrec; ++ir) {
        intp[ir] += slc1[ir]*ls1[is];
        intp[ir] += slc2[ir]*ls2[is];
      }
      /* Save to output */
      memcpy(&datf[k*nrec],intp,sizeof(float)*nrec);
      for(int ir = 0; ir < nrec; ++ir) intp[ir] = 0;
      ++k;
    }
  }

  /* Free memory */
  delete[] slc1; delete[] slc2; delete[] intp;
}

void scaas2d::fwdprop_oneshot(float *src, int *srcxs, int *srczs, int nsrc, int *recxs, int *reczs, int nrec, float *vel, float *dat) {

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
    sou1[srczs[isrc]*_nx + srcxs[isrc]] = src[isrc*_ntu + 0]/(_dx*_dz);
  }
  for(int k = 0; k < _onestp; ++k) { sou1[k] *= v2dt2[k]; };
  apply_taper(tap,sou1,pre);
  drslc(recxs, reczs, nrec, sou1, dslc);
  memcpy(&dat[nrec],dslc,sizeof(float)*nrec); // p(x,1) = v^2*dt^2*f(0)
  memcpy(cur,sou1,sizeof(float)*_onestp);

  int kt = 1;
  if(_skip == 1) kt = 2; // Handle the special case
  for(int it = 2; it < _ntu; ++it) {
    /* Calculate source term */
    for(int isrc = 0; isrc < nsrc; ++isrc) {
      sou1[srczs[isrc]*_nx + srcxs[isrc]] = src[isrc*_ntu + (it-1)]/(_dx*_dz);
    }
    /* Apply laplacian */
    laplacian10(_nx,_nz,_idx2,_idz2,cur,sou2);
    /* Advance wavefields */
    for(int k = 0; k < _onestp; ++k) {
      pre[k] = v2dt2[k]*sou1[k] + 2*cur[k] + v2dt2[k]*sou2[k] - pre[k];
    }
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

void scaas2d::fwdprop_wfld(float *src, int *srcxs, int *srczs, int nsrc, float *vel, float *psol) {
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
  float *tmp;

  /* Initial conditions */
  memcpy(psol,pre,sizeof(float)*_onestp); //p(x,0) = 0.0

  /* Inject sources */
  for(int isrc = 0; isrc < nsrc; ++isrc) {
    sou1[srczs[isrc]*_nx + srcxs[isrc]] = src[isrc*_ntu + 0]/(_dx*_dz);
  }
  for(int k = 0; k < _onestp; ++k) { sou1[k] *= v2dt2[k]; };
  apply_taper(tap,sou1,pre);
  memcpy(cur,sou1,sizeof(float)*_onestp);

  int kt = 1;
  if(_skip == 1) kt = 2; // Handle the special case
  for(int it = 2; it < _ntu; ++it) {
    /* Calculate source term */
    for(int isrc = 0; isrc < nsrc; ++isrc) {
      sou1[srczs[isrc]*_nx + srcxs[isrc]] = src[isrc*_ntu + (it-1)]/(_dx*_dz);
    }
    /* Apply laplacian */
    laplacian10(_nx,_nz,_idx2,_idz2,cur,sou2);
    /* Advance wavefields */
    for(int k = 0; k < _onestp; ++k) {
      pre[k] = v2dt2[k]*sou1[k] + 2*cur[k] + v2dt2[k]*sou2[k] - pre[k];
    }
    /* Apply taper */
    apply_taper(tap,pre,cur);
    /* Save wavefield on coarse time grid */
    if(it%_skip == 0) {
      /* Copy to wavefield vector */
      memcpy(&psol[kt*_onestp],pre,sizeof(float)*_onestp);
      kt++;
    }
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Free memory */
  delete[] v2dt2;
  delete[] pre; delete[] cur; delete[] sou1;
  delete[] sou2; delete[] tap;

}

void scaas2d::fwdprop_lapwfld(float *src, int *srcxs, int *srczs, int nsrc, float *vel, float *lappsol) {
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
  float *tmp;

  /* Initial conditions */
  memcpy(lappsol,pre,sizeof(float)*_onestp); //p(x,0) = 0.0

  /* Inject sources */
  for(int isrc = 0; isrc < nsrc; ++isrc) {
    sou1[srczs[isrc]*_nx + srcxs[isrc]] = src[isrc*_ntu + 0]/(_dx*_dz);
  }
  for(int k = 0; k < _onestp; ++k) { sou1[k] *= v2dt2[k]; };
  apply_taper(tap,sou1,pre);
  memcpy(cur,sou1,sizeof(float)*_onestp);

  int kt = 1;
  for(int it = 2; it < _ntu; ++it) {
    /* Calculate source term */
    for(int isrc = 0; isrc < nsrc; ++isrc) {
      sou1[srczs[isrc]*_nx + srcxs[isrc]] = src[isrc*_ntu + (it-1)]/(_dx*_dz);
    }
    /* Apply laplacian */
    laplacian10(_nx,_nz,_idx2,_idz2,cur,sou2);
    /* Advance wavefields */
    for(int k = 0; k < _onestp; ++k) {
      pre[k] = v2dt2[k]*sou1[k] + 2*cur[k] + v2dt2[k]*sou2[k] - pre[k];
    }
    /* Apply taper */
    apply_taper(tap,pre,cur);
    /* Save wavefield on coarse time grid */
    if((it-1)%_skip == 0) {
      /* Copy to wavefield vector */
      memcpy(&lappsol[kt*_onestp],sou2,sizeof(float)*_onestp);
      kt++;
    }
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Free memory */
  delete[] v2dt2;
  delete[] pre; delete[] cur; delete[] sou1;
  delete[] sou2; delete[] tap;

}

void scaas2d::fwdprop_multishot(float *src, int *srcxs, int *srczs, int *nsrcs, int *recxs, int *reczs, int *nrecs, int nex,
    float *vel, float *dat, int nthrds) {

  /* Precompute the offsets */
  int *soffsets = new int[nex](); int *roffsets = new int[nex]();
  for(int iex = 1; iex < nex; ++iex) {
    soffsets[iex] = soffsets[iex-1] + nsrcs[iex];
    roffsets[iex] = roffsets[iex-1] + nrecs[iex];
  }

  /* Loop over each experiment */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iex = 0; iex < nex; ++iex) {
    /* Get number of sources and receivers for this shot */
    int insrc = nsrcs[iex]; int inrec = nrecs[iex];
    /* Get the source positions for this shot */
    int *isrcx = new int[insrc](); int *isrcz = new int[insrc]();
    memcpy(isrcx,&srcxs[soffsets[iex]],sizeof(int)*insrc); memcpy(isrcz,&srczs[soffsets[iex]],sizeof(int)*insrc);
    /* Get the receiver positions for this shot */
    int *irecx = new int[inrec](); int *irecz = new int[inrec]();
    memcpy(irecx,&recxs[roffsets[iex]],sizeof(int)*inrec); memcpy(irecz,&reczs[roffsets[iex]],sizeof(int)*inrec);
    /* Get the source wavelets for this shot */
    float *isrc = new float[_ntu*insrc]();
    memcpy(isrc,&src[soffsets[iex]*_ntu],sizeof(float)*_ntu*insrc);
    /* Perform the wave propagation for this shot */
    float *idat = new float[_nt*inrec]();
    fwdprop_oneshot(isrc, isrcx, isrcz, insrc, irecx, irecz, inrec, vel, idat);
    /* Copy to output data array */
    memcpy(&dat[iex*_nt*inrec],idat,sizeof(float)*_nt*inrec);
    /* Free memory */
    delete[] isrcx;  delete[] isrcz;
    delete[] irecx;  delete[] irecz;
    delete[] isrc; delete[] idat;
  }
  delete[] soffsets; delete[] roffsets;
}

void scaas2d::adjprop_wfld(float *asrc, int *recxs, int *reczs, int nrec, float *vel, float *lsol) {

  /* Interpolate adjoint source onto fine time grid */
  float *iasrc = new float[_ntu*nrec]();
  shot_interp(nrec,asrc,iasrc);

  /* Precompute velocity dt^2 coefficient */
  float *v2dt2 = new float[_onestp]();
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
  float *tmp;

  /* First terminal condition (l(nx,nt-1) = 0.0) */
  memcpy(&lsol[_onestp*(_nt-1)],pre,sizeof(float)*_onestp);

  /* Second step, inject adjoint source at next final step */
  for(int irec = 0; irec < nrec; ++irec) {
    sou1[reczs[irec]*_nx + recxs[irec]] = iasrc[(_ntu-1)*nrec + irec];
  }
  for(int k = 0; k < _onestp; ++k) { sou1[k] *= v2dt2[k]; };
  apply_taper(tap,sou1,pre);
  memcpy(cur,sou1,sizeof(float)*_onestp);

  int kt = _nt-1;
  if(_skip == 1) kt = _nt-3; //Handle the special case
  for(int it = _ntu-3; it > -1; --it) {
    /* Inject adjoint source */
    for(int irec = 0; irec < nrec; ++irec) {
      sou1[reczs[irec]*_nx + recxs[irec]] = iasrc[(it+1)*nrec + irec]/(_dx*_dz);
    }
    /* Apply laplacian */
    laplacian10(_nx,_nz,_idx2,_idz2,cur,sou2);
    /* Advance wavefields */
    for(int k = 0; k < _onestp; ++k) {
      pre[k] = v2dt2[k]*sou1[k] + 2*cur[k] + v2dt2[k]*sou2[k] - pre[k];
    }
    /* Apply taper */
    apply_taper(tap,pre,cur);
    /* Accumulate gradient on coarse time grid */
    if((it)%_skip == 0) {
      /* Copy to wavefield vector */
      memcpy(&lsol[kt*_onestp],pre,sizeof(float)*_onestp);
      kt--;
    }
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Free memory */
  delete[] iasrc;
  delete[] v2dt2;  delete[] tap;
  delete[] pre; delete[] cur; delete[] sou1;
  delete[] sou2;
}

void scaas2d::d2t(float* p ,float *d2p) {
  /* Temporal sampling */
  float idt2 = 1/(_dt * _dt);

  /* Initial rows */
  for(int k = 0; k < _onestp; ++k) { d2p[k] = p[k]*idt2; }
  for(int k = 0; k < _onestp; ++k) {
    d2p[_onestp + k] = (p[_onestp + k] - 2*p[k])*idt2;
  }

  for(int it = 2; it < _nt; ++it) {
    for(int k = 0; k < _onestp; ++k) {
      d2p[it*_onestp + k] = (p[it*_onestp + k] - 2*p[(it-1)*_onestp + k] + p[(it-2)*_onestp + k])*idt2;
    }
  }
}

void scaas2d::d2x(float *p, float *d2p) {
  /* Allocate temporary arrays */
  float *tmp1 = new float[_onestp]();
  float *tmp2 = new float[_onestp]();

  /* Laplacian slice by slice */
  for(int it = 0; it < _nt; ++it) {
    memcpy(tmp1,&p[it*_onestp],sizeof(float)*_onestp);
    laplacian10(_nx,_nz,_idx2,_idz2,tmp1,tmp2);
    memcpy(&d2p[it*_onestp],tmp2,sizeof(float)*_onestp);
  }

  /* Free memory */
  delete[] tmp1; delete[] tmp2;

}

void scaas2d::lapimg(float *img, float *lap) {

  for(int iz = 0; iz < _nz; ++iz) {
    for(int ix = 0; ix < _nx; ++ix) {
      lap[iz*_nx + ix] = -4*img[iz*_nx + ix] +
          img[(iz-1)*_nx + ix  ] + img[(iz+1)*_nx + ix  ] +
          img[(iz  )*_nx + ix-1] + img[(iz  )*_nx + ix+1];
    }
  }

}

void scaas2d::calc_grad_d2t(float *d2pt, float *lsol, float *v, float *grad) {

  /* Velocity coefficient */
  float *ivel3 = new float[_onestp]();

  /* Precompute velocity term */
  for(int k = 0; k < _onestp; ++k) {
    if(v[k] != 0.0) {
      ivel3[k] = -2.0/(v[k]*v[k]*v[k]);
    }
  }

  /* Gradient computation */
  for(int it = 1; it < _nt; ++it) {
    for(int k = 0; k < _onestp; ++k) {
      grad[k] += ivel3[k] * (lsol[_onestp*(it-1) + k] * d2pt[_onestp*it + k])*(_dx*_dz);
    }
  }

  /* Free memory */
  delete[] ivel3;

}

void scaas2d::calc_grad_d2x(float *d2px, float *lsol, float *src, int *srcxs, int *srczs, int nsrc, float *v, float *grad) {

  /* Temporary arrays */
  float *ivel = new float[_onestp]();
  float *tsrc = new float[_onestp]();

  /* Precompute velocity term */
  for(int k = 0; k < _onestp; ++k) {
    if(v[k] != 0.0) {
      ivel[k] = -2/v[k];
    }
  }

  /* Gradient computation */
  for(int it = 0; it < _nt; ++it){
    /* Inject source */
    for(int isrc = 0; isrc < nsrc; ++isrc) {
      tsrc[srczs[isrc]*_nx + srcxs[isrc]] = src[isrc*_ntu + it]/(_dx*_dz);
    }
    /* Gradient formula */
    for(int k = 0; k < _onestp; ++k) {
      grad[k] += ivel[k] * (lsol[_onestp*it + k] * d2px[_onestp*it + k] + lsol[_onestp*it + k] * tsrc[k])*(_dx*_dz);
    }
  }

  /* Free memory */
  delete[] ivel; delete[] tsrc;

}

void scaas2d::gradient_oneshot(float *src, int *srcxs, int *srczs, int nsrc, float *asrc, int *recxs, int *reczs, int nrec, float *vel, float *grad) {

  /* First calculate laplacian of background wavefield */
  float *lappsol = new float[_nt*_onestp]();
  fwdprop_lapwfld(src, srcxs, srczs, nsrc, vel, lappsol);

  /* Subsample the source wavelet */
  int kts = 0;
  float *srccrse = new float[_nt]();
  for(int it = 0; it < _ntu; ++it) {
    if(it%_skip == 0) {
      srccrse[kts] = src[it];
      kts++;
    }
  }

  /* Precompute velocity term */
  //TODO: Work in log!
  float *ivel = new float[_onestp]();
  for(int k = 0; k < _onestp; ++k) {
    if(vel[k] != 0.0) {
      ivel[k] = -2.0/vel[k];
    }
  }

  /* Interpolate adjoint source onto fine time grid */
  float *iasrc = new float[_ntu*nrec]();
  shot_interp(nrec,asrc,iasrc);

  /* Precompute velocity dt^2 coefficient */
  float *v2dt2 = new float[_onestp]();
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
  float *tsrc = new float[_onestp]();
  float *tmp;

  /* First terminal condition (l(nx,nt-1) = 0.0). Do nothing here */

  /* Second step, inject adjoint source at next final step */
  for(int irec = 0; irec < nrec; ++irec) {
    sou1[reczs[irec]*_nx + recxs[irec]] = iasrc[(_ntu-1)*nrec + irec]/(_dx*_dz);
  }
  for(int k = 0; k < _onestp; ++k) { sou1[k] *= v2dt2[k]; };
  apply_taper(tap,sou1,pre);
  memcpy(cur,sou1,sizeof(float)*_onestp);

  int kt = _nt-1;
  if(_skip == 1) kt = _ntu-3; // Handle the special case
  for(int it = _ntu-3; it > -1; --it) {
    /* Inject adjoint source */
    for(int irec = 0; irec < nrec; ++irec) {
      sou1[reczs[irec]*_nx + recxs[irec]] = iasrc[(it+1)*nrec + irec]/(_dx*_dz);
    }
    /* Apply laplacian */
    laplacian10(_nx,_nz,_idx2,_idz2,cur,sou2);
    /* Advance wavefields */
    for(int k = 0; k < _onestp; ++k) {
      pre[k] = v2dt2[k]*sou1[k] + 2*cur[k] + v2dt2[k]*sou2[k] - pre[k];
    }
    /* Apply taper */
    apply_taper(tap,pre,cur);
    /* Accumulate gradient on coarse time grid */
    if((it)%_skip == 0) {
      /* Inject the forward source */
      for(int isrc = 0; isrc < nsrc; ++isrc) {
        tsrc[srczs[isrc]*_nx + srcxs[isrc]] = srccrse[isrc*_ntu + kt]/(_dx*_dz);
      }
      /* Gradient formula */
      for(int k = 0; k < _onestp; ++k) {
        grad[k] += ivel[k] * (pre[k] * lappsol[kt*_onestp + k] + pre[k]*tsrc[k])*(_dx*_dz);
      }
      kt--;
    }
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Free memory */
  delete[] lappsol; delete[] iasrc;
  delete[] v2dt2;  delete[] tap;
  delete[] pre; delete[] cur; delete[] sou1;
  delete[] sou2; delete[] tsrc;
  delete[] ivel; delete[] srccrse;
}


void scaas2d::gradient_multishot(float *src, int *srcxs, int *srczs, int *nsrcs, float *asrc, int *recxs, int *reczs, int *nrecs, int nex,
    float *vel, float *grad, int nthrds) {

  /* Precompute the offsets */
  int *soffsets = new int[nex](); int *roffsets = new int[nex]();
  for(int iex = 1; iex < nex; ++iex) {
    soffsets[iex] = soffsets[iex-1] + nsrcs[iex];
    roffsets[iex] = roffsets[iex-1] + nrecs[iex];
  }

  /* Loop over each experiment */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iex = 0; iex < nex; ++iex) {
    /* Get number of sources and receivers for this shot */
    int insrc = nsrcs[iex]; int inrec = nrecs[iex];
    /* Get the source positions for this shot */
    int *isrcx = new int[insrc](); int *isrcz = new int[insrc]();
    memcpy(isrcx,&srcxs[soffsets[iex]],sizeof(int)*insrc); memcpy(isrcz,&srczs[soffsets[iex]],sizeof(int)*insrc);
    /* Get the receiver positions for this shot */
    int *irecx = new int[inrec](); int *irecz = new int[inrec]();
    memcpy(irecx,&recxs[roffsets[iex]],sizeof(int)*inrec); memcpy(irecz,&reczs[roffsets[iex]],sizeof(int)*inrec);
    /* Get the source wavelets for this shot */
    float *isrc = new float[_ntu*insrc]();
    memcpy(isrc,&src[soffsets[iex]*_ntu],sizeof(float)*_ntu*insrc);
    /* Get the adjoint sources for this shot */
    float *iasrc = new float[_nt*inrec]();
    memcpy(iasrc,&asrc[iex*_nt*inrec],sizeof(float)*_nt*inrec);
    /* Allocate a temporary gradient */
    float *igrad = new float[_onestp]();
    /* Compute the gradient for this shot */
    gradient_oneshot(isrc, isrcx, isrcz, insrc, iasrc, irecx, irecz, inrec, vel, igrad);
    /* Add gradient to output gradient */
    for(int k = 0; k < _onestp; ++k) { grad[k] += igrad[k]; };
    /* Free memory */
    delete[] isrcx;  delete[] isrcz;
    delete[] irecx;  delete[] irecz;
    delete[] isrc;   delete[] iasrc;
    delete[] igrad;
  }
  delete[] soffsets; delete[] roffsets;
}

void scaas2d::brnfwd_oneshot(float *src, int *srcxs, int *srczs, int nsrc, int *recxs, int *reczs, int nrec, float *vel, float *dvel, float *ddat) {
  /* Precompute velocity dt^2 coefficient */
  float *v2dt2 = new float[_onestp]();
  //TODO: This should be done outside
  for(int k = 0; k < _onestp; ++k) { v2dt2[k] = vel[k]*vel[k]*_dtu*_dtu; };

  /* Precompute Born source scale factor */
  float *dvov = new float[_onestp]();
  for(int k = 0; k < _onestp; ++k) {
    if(vel[k] != 0) dvov[k] = 2*dvel[k]/vel[k];
  }

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  float *tap = new float[_onestp]();
  build_taper(tap);

  /* Allocate memory for wavefield slices */
  float *pre  = new float[_onestp]();  float *dpre  = new float[_onestp]();
  float *cur  = new float[_onestp]();  float *dcur  = new float[_onestp]();
  float *sou1 = new float[_onestp]();  float *dsou1 = new float[_onestp]();
  float *sou2 = new float[_onestp]();  float *dsou2 = new float[_onestp]();
  float *ddslc = new float[_nt*nrec]();
  float *tmp;                          float *dtmp;

  /* Background initial conditions */
  for(int isrc = 0; isrc < nsrc; ++isrc) {
    sou1[srczs[isrc]*_nx + srcxs[isrc]] = src[isrc*_ntu + 0]/(_dx*_dz); // Source injection
  }
  for(int k = 0; k < _onestp; ++k) { sou1[k] *= v2dt2[k]; }; // p(x,1) = v^2*dt^2*f(0)
  memcpy(cur,sou1,sizeof(float)*_onestp);

  /* Scattered initial conditions */
  memcpy(ddat,ddslc,sizeof(float)*nrec); //dp(x,0) = 0.0
  for(int k = 0; k < _onestp; ++k) { dcur[k] = dvov[k]*sou1[k]; }; // dp(x,1) = dv/v*(lap(p(x,0)) + f(0))


  int kt = 1;
  if(_skip == 1) kt = 2; // Handle the special case
  for(int it = 2; it < _ntu; ++it) {

    /* First compute the background wavefield */
    // Calculate source term
    for(int isrc = 0; isrc < nsrc; ++isrc) {
      sou1[srczs[isrc]*_nx + srcxs[isrc]] = src[isrc*_ntu + (it-1)]/(_dx*_dz);
    }
    // Apply laplacian
    laplacian10(_nx,_nz,_idx2,_idz2,cur,sou2);
    // Advance background wavefield
    for(int k = 0; k < _onestp; ++k) {
      pre[k] = v2dt2[k]*sou1[k] + 2*cur[k] + v2dt2[k]*sou2[k] - pre[k];
    }
    /* Apply taper */
    apply_taper(tap,pre,cur);

    /* Now compute scattered wavefield using the Born source */
    // Apply laplacian
    laplacian10(_nx,_nz,_idx2,_idz2,dcur,dsou2);
    for(int k = 0; k < _onestp; ++k) {
      dpre[k] = v2dt2[k]*(dvov[k]*sou2[k] +  dvov[k]*sou1[k]) + 2*dcur[k] + v2dt2[k]*dsou2[k] - dpre[k];
    }
    /* Apply taper */
    apply_taper(tap,dpre,dcur);

    /* Save scattered data on coarse time grid */
    if(it%_skip == 0) {
      /* Apply delta function operator */
      drslc(recxs,reczs,nrec,dpre,ddslc);
      /* Copy to data vector */
      memcpy(&ddat[kt*nrec],ddslc,sizeof(float)*nrec);
      kt++;
    }
    /* Background pointer swap */
    tmp = cur; cur = pre; pre = tmp;
    /* Scattered pointer swap */
    dtmp = dcur; dcur = dpre; dpre = dtmp;
  }

  /* Free memory */
  delete[] v2dt2;
  delete[] pre; delete[] cur; delete[] sou1;
  delete[] sou2; delete[] tap;
  delete[] dpre; delete[] dcur; delete[] dsou1;
  delete[] dsou2; delete[] ddslc;
}

void scaas2d::brnfwd(float *src, int *srcxs, int *srczs, int *nsrcs, int *recxs, int *reczs, int *nrecs, int nex,
    float *vel, float *dvel, float *ddat, int nthrds) {
  /* Precompute the offsets */
  int *soffsets = new int[nex](); int *roffsets = new int[nex]();
  for(int iex = 1; iex < nex; ++iex) {
    soffsets[iex] = soffsets[iex-1] + nsrcs[iex];
    roffsets[iex] = roffsets[iex-1] + nrecs[iex];
  }

  /* Loop over each experiment */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iex = 0; iex < nex; ++iex) {
    /* Get number of sources and receivers for this shot */
    int insrc = nsrcs[iex]; int inrec = nrecs[iex];
    /* Get the source positions for this shot */
    int *isrcx = new int[insrc](); int *isrcz = new int[insrc]();
    memcpy(isrcx,&srcxs[soffsets[iex]],sizeof(int)*insrc); memcpy(isrcz,&srczs[soffsets[iex]],sizeof(int)*insrc);
    /* Get the receiver positions for this shot */
    int *irecx = new int[inrec](); int *irecz = new int[inrec]();
    memcpy(irecx,&recxs[roffsets[iex]],sizeof(int)*inrec); memcpy(irecz,&reczs[roffsets[iex]],sizeof(int)*inrec);
    /* Get the source wavelets for this shot */
    float *isrc = new float[_ntu*insrc]();
    memcpy(isrc,&src[soffsets[iex]*_ntu],sizeof(float)*_ntu*insrc);
    /* Perform the wave propagation for this shot */
    float *iddat = new float[_nt*inrec]();
    brnfwd_oneshot(isrc, isrcx, isrcz, insrc, irecx, irecz, inrec, vel, dvel, iddat);
    /* Copy to output data array */
    memcpy(&ddat[iex*_nt*inrec],iddat,sizeof(float)*_nt*inrec);
    /* Free memory */
    delete[] isrcx;  delete[] isrcz;
    delete[] irecx;  delete[] irecz;
    delete[] isrc; delete[] iddat;
  }
  delete[] soffsets; delete[] roffsets;
}

void scaas2d::brnadj_oneshot(float *src, int *srcxs, int *srczs, int nsrc, int *recxs, int *reczs, int nrec, float *vel, float *dv, float *ddat) {

  /* First calculate laplacian of background wavefield */
  float *lappsol = new float[_nt*_onestp]();
  fwdprop_lapwfld(src, srcxs, srczs, nsrc, vel, lappsol);

  /* Subsample the source wavelet */
  int kts = 0;
  float *srccrse = new float[_nt]();
  for(int it = 0; it < _ntu; ++it) {
    if(it%_skip == 0) {
      srccrse[kts] = src[it];
      kts++;
    }
  }

  /* Precompute velocity term */
  float *ivel = new float[_onestp]();
  for(int k = 0; k < _onestp; ++k) {
    if(vel[k] != 0.0) {
      ivel[k] = -2.0/vel[k];
    }
  }

  /* Interpolate adjoint source onto fine time grid */
  float *iddat = new float[_ntu*nrec]();
  shot_interp(nrec,ddat,iddat);

  /* Precompute velocity dt^2 coefficient */
  float *v2dt2 = new float[_onestp]();
  for(int k = 0; k < _onestp; ++k) { v2dt2[k] = vel[k]*vel[k]*_dtu*_dtu; };

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  float *tap = new float[_onestp]();
  build_taper(tap);

  /* Allocate memory for wavefield slices */
  float *pre  = new float[_onestp]();
  float *cur  = new float[_onestp]();
  float *sou1 = new float[_onestp]();
  float *sou2 = new float[_onestp]();
  float *tsrc = new float[_onestp]();
  float *tmp;

  /* First terminal condition (l(nx,nt-1) = 0.0). Do nothing here */

  /* Second step, inject adjoint source at next final step */
  for(int irec = 0; irec < nrec; ++irec) {
    sou1[reczs[irec]*_nx + recxs[irec]] = -iddat[(_ntu-1)*nrec + irec]/(_dx*_dz);
  }
  for(int k = 0; k < _onestp; ++k) { sou1[k] *= v2dt2[k]; };
  apply_taper(tap,sou1,pre);
  memcpy(cur,sou1,sizeof(float)*_onestp);

  int kt = _nt-1;
  if(_skip == 1) kt = _ntu-3; // Handle the special case
  for(int it = _ntu-3; it > -1; --it) {
    /* Inject adjoint source */
    for(int irec = 0; irec < nrec; ++irec) {
      sou1[reczs[irec]*_nx + recxs[irec]] = -iddat[(it+1)*nrec + irec]/(_dx*_dz);
    }
    /* Apply laplacian */
    laplacian10(_nx,_nz,_idx2,_idz2,cur,sou2);
    /* Advance wavefields */
    for(int k = 0; k < _onestp; ++k) {
      pre[k] = v2dt2[k]*sou1[k] + 2*cur[k] + v2dt2[k]*sou2[k] - pre[k];
    }
    /* Apply taper */
    apply_taper(tap,pre,cur);
    /* Accumulate gradient on coarse time grid */
    if((it)%_skip == 0) {
      /* Inject the forward source */
      for(int isrc = 0; isrc < nsrc; ++isrc) {
        tsrc[srczs[isrc]*_nx + srcxs[isrc]] = srccrse[isrc*_ntu + kt]/(_dx*_dz);
      }
      /* Gradient formula */
      for(int k = 0; k < _onestp; ++k) {
        dv[k] += ivel[k] * (pre[k] * lappsol[kt*_onestp + k] + pre[k]*tsrc[k])*(_dx*_dz);
      }
      kt--;
    }
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Free memory */
  delete[] lappsol; delete[] iddat;
  delete[] v2dt2;  delete[] tap;
  delete[] pre; delete[] cur; delete[] sou1;
  delete[] sou2; delete[] tsrc;
  delete[] ivel; delete[] srccrse;
}

void scaas2d::brnadj(float *src, int *srcxs, int *srczs, int *nsrcs, int *recxs, int *reczs, int *nrecs, int nex,
    float *vel, float *dvel, float *ddat, int nthrds) {

  /* Precompute the offsets */
  int *soffsets = new int[nex](); int *roffsets = new int[nex]();
  for(int iex = 1; iex < nex; ++iex) {
    soffsets[iex] = soffsets[iex-1] + nsrcs[iex];
    roffsets[iex] = roffsets[iex-1] + nrecs[iex];
  }

  /* Loop over each experiment */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iex = 0; iex < nex; ++iex) {
    /* Get number of sources and receivers for this shot */
    int insrc = nsrcs[iex]; int inrec = nrecs[iex];
    /* Get the source positions for this shot */
    int *isrcx = new int[insrc](); int *isrcz = new int[insrc]();
    memcpy(isrcx,&srcxs[soffsets[iex]],sizeof(int)*insrc); memcpy(isrcz,&srczs[soffsets[iex]],sizeof(int)*insrc);
    /* Get the receiver positions for this shot */
    int *irecx = new int[inrec](); int *irecz = new int[inrec]();
    memcpy(irecx,&recxs[roffsets[iex]],sizeof(int)*inrec); memcpy(irecz,&reczs[roffsets[iex]],sizeof(int)*inrec);
    /* Get the source wavelets for this shot */
    float *isrc = new float[_ntu*insrc]();
    memcpy(isrc,&src[soffsets[iex]*_ntu],sizeof(float)*_ntu*insrc);
    /* Get the data for this shot */
    float *iddat = new float[_nt*inrec]();
    memcpy(iddat,&ddat[iex*_nt*inrec],sizeof(float)*_nt*inrec);
    /* Allocate a temporary gradient */
    float *idvel = new float[_onestp]();
    /* Compute the gradient for this shot */
    brnadj_oneshot(isrc, isrcx, isrcz, insrc, irecx, irecz, inrec, vel, idvel, iddat);
    /* Add gradient to output gradient */
    for(int k = 0; k < _onestp; ++k) { dvel[k] += idvel[k]; };
    /* Free memory */
    delete[] isrcx;  delete[] isrcz;
    delete[] irecx;  delete[] irecz;
    delete[] isrc;   delete[] iddat;
    delete[] idvel;
  }
  delete[] soffsets; delete[] roffsets;
}

void scaas2d::brnoffadj_oneshot(float *src, int *srcxs, int *srczs, int nsrc, int *recxs, int *reczs, int nrec, float *vel, int rnh, float *dv, float *ddat) {

  /* First calculate laplacian of background wavefield */
  float *lappsol = new float[_nt*_onestp]();
  fwdprop_lapwfld(src, srcxs, srczs, nsrc, vel, lappsol);

  /* Subsample the source wavelet */
  int kts = 0;
  float *srccrse = new float[_nt]();
  for(int it = 0; it < _ntu; ++it) {
    if(it%_skip == 0) {
      srccrse[kts] = src[it];
      kts++;
    }
  }

  /* Precompute velocity term */
  float *ivel = new float[_onestp]();
  for(int k = 0; k < _onestp; ++k) {
    if(vel[k] != 0.0) {
      ivel[k] = -2.0/vel[k];
    }
  }

  /* Compute extended image axis */
  int nh = (rnh - 1)/2;

  /* Interpolate adjoint source onto fine time grid */
  float *iddat = new float[_ntu*nrec]();
  shot_interp(nrec,ddat,iddat);

  /* Precompute velocity dt^2 coefficient */
  float *v2dt2 = new float[_onestp]();
  for(int k = 0; k < _onestp; ++k) { v2dt2[k] = vel[k]*vel[k]*_dtu*_dtu; };

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  float *tap = new float[_onestp]();
  build_taper(tap);

  /* Allocate memory for wavefield slices */
  float *pre  = new float[_onestp]();
  float *cur  = new float[_onestp]();
  float *sou1 = new float[_onestp]();
  float *sou2 = new float[_onestp]();
  float *tsrc = new float[_onestp]();
  float *tmp;

  /* First terminal condition (l(nx,nt-1) = 0.0). Do nothing here */

  /* Second step, inject adjoint source at next final step */
  for(int irec = 0; irec < nrec; ++irec) {
    sou1[reczs[irec]*_nx + recxs[irec]] = -iddat[(_ntu-1)*nrec + irec]/(_dx*_dz);
  }
  for(int k = 0; k < _onestp; ++k) { sou1[k] *= v2dt2[k]; };
  apply_taper(tap,sou1,pre);
  memcpy(cur,sou1,sizeof(float)*_onestp);

  int kt = _nt-1;
  if(_skip == 1) kt = _ntu-3; // Handle the special case
  for(int it = _ntu-3; it > -1; --it) {
    /* Inject adjoint source */
    for(int irec = 0; irec < nrec; ++irec) {
      sou1[reczs[irec]*_nx + recxs[irec]] = -iddat[(it+1)*nrec + irec]/(_dx*_dz);
    }
    /* Apply laplacian */
    laplacian10(_nx,_nz,_idx2,_idz2,cur,sou2);
    /* Advance wavefields */
    for(int k = 0; k < _onestp; ++k) {
      pre[k] = v2dt2[k]*sou1[k] + 2*cur[k] + v2dt2[k]*sou2[k] - pre[k];
    }
    /* Apply taper */
    apply_taper(tap,pre,cur);
    /* Accumulate gradient on coarse time grid */
    if((it)%_skip == 0) {
      /* Inject the forward source */
      for(int isrc = 0; isrc < nsrc; ++isrc) {
        tsrc[srczs[isrc]*_nx + srcxs[isrc]] = srccrse[isrc*_ntu + kt]/(_dx*_dz);
      }
      /* Extended gradient formula */
      for(int ih = -nh; ih <= nh; ++ih) {
        for(int iz = 0; iz < _nz; ++iz) {
          for(int ix = nh; ix < _nx-nh; ++ix) {
            dv[(ih+nh)*_onestp + iz*_nx + ix] += ivel[iz*_nx + ix] * (pre[iz*_nx + ix+ih] * lappsol[kt*_onestp + iz*_nx + ix-ih] +
                pre[iz*_nx + ix+ih] * tsrc[iz*_nx + ix-ih])*(_dx*_dz);
          }
        }
      }
      kt--;
    }
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Free memory */
  delete[] lappsol; delete[] iddat;
  delete[] v2dt2;  delete[] tap;
  delete[] pre; delete[] cur; delete[] sou1;
  delete[] sou2; delete[] tsrc;
  delete[] ivel; delete[] srccrse;
}

void scaas2d::brnoffadj(float *src, int *srcxs, int *srczs, int *nsrcs, int *recxs, int *reczs, int *nrecs, int nex,
    float *vel, int rnh, float *dvel, float *ddat, int nthrds) {

  /* Precompute the offsets */
  int *soffsets = new int[nex](); int *roffsets = new int[nex]();
  for(int iex = 1; iex < nex; ++iex) {
    soffsets[iex] = soffsets[iex-1] + nsrcs[iex];
    roffsets[iex] = roffsets[iex-1] + nrecs[iex];
  }

  /* Loop over each experiment */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iex = 0; iex < nex; ++iex) {
    /* Get number of sources and receivers for this shot */
    int insrc = nsrcs[iex]; int inrec = nrecs[iex];
    /* Get the source positions for this shot */
    int *isrcx = new int[insrc](); int *isrcz = new int[insrc]();
    memcpy(isrcx,&srcxs[soffsets[iex]],sizeof(int)*insrc); memcpy(isrcz,&srczs[soffsets[iex]],sizeof(int)*insrc);
    /* Get the receiver positions for this shot */
    int *irecx = new int[inrec](); int *irecz = new int[inrec]();
    memcpy(irecx,&recxs[roffsets[iex]],sizeof(int)*inrec); memcpy(irecz,&reczs[roffsets[iex]],sizeof(int)*inrec);
    /* Get the source wavelets for this shot */
    float *isrc = new float[_ntu*insrc]();
    memcpy(isrc,&src[soffsets[iex]*_ntu],sizeof(float)*_ntu*insrc);
    /* Get the adjoint sources for this shot */
    float *iddat = new float[_nt*inrec]();
    memcpy(iddat,&ddat[iex*_nt*inrec],sizeof(float)*_nt*inrec);
    /* Allocate a temporary gradient */
    float *idvel = new float[_onestp*rnh]();
    /* Compute the extended image for this shot */
    brnoffadj_oneshot(isrc, isrcx, isrcz, insrc, irecx, irecz, inrec, vel, rnh, idvel, iddat);
    /* Add image to output image */
    for(int k = 0; k < _onestp*rnh; ++k) { dvel[k] += idvel[k]; };
    /* Free memory */
    delete[] isrcx;  delete[] isrcz;
    delete[] irecx;  delete[] irecz;
    delete[] isrc;   delete[] iddat;
    delete[] idvel;
  }
  delete[] soffsets; delete[] roffsets;
}

void scaas2d::build_taper(float *tap) {
  /* Fill the the taper with ones */
  for(int k = 0; k < _onestp; ++k) { tap[k] = 1.0; };

  /* Always use 10th order */
  int ord = 10;
  int shift = ord/2;

  /* Top and Bottom */
  for(int iz = shift; iz < _bz+shift; ++iz) {
    for(int ix = shift; ix < _nx-shift; ++ix) {
      double tapval = 1.0 * (_bz - (iz - shift))/(_bz);
      tapval = _alpha + (1.0 - _alpha) * cos(M_PI*tapval);
      tap[iz        *_nx + ix] = tapval;
      tap[(_nz-iz-1)*_nx + ix] = tapval;
    }
  }

  /* Left and Right */
  for(int ix = shift; ix < _bx+shift; ++ix) {
    for(int iz = shift; iz < _nz-shift; ++iz) {
      double tapval = 1.0 * (_bx - (ix - shift))/(_bx);
      tapval = _alpha + (1.0 - _alpha) * cos(M_PI*tapval);
      tap[iz*_nx + ix      ] *= tapval;
      tap[iz*_nx + _nx-ix-1] *= tapval;
    }
  }
}

void scaas2d::apply_taper(float *tap, float *cur, float *nex) {

  /* Always use 10th order */
  int ord = 10;
  int shift = ord/2;
  int nshift = (_nz-_bz-shift)*_nx;

  /* Top and Bottom application */
  for(int itop = shift; itop < (_bz+shift)*(_nx); ++itop) {
    int ibot = nshift + itop;
    cur[itop] *= tap[itop];
    nex[itop] *= tap[itop];
    cur[ibot] *= tap[ibot];
    nex[ibot] *= tap[ibot];
  }

  /* Left and right application */
  for(int iz = _bz+shift; iz < _nz-_bz-shift; ++iz) {
    for(int ix = shift; ix < _bx+shift; ++ix) {
      cur[iz*_nx +       ix] *= tap[iz*_nx +       ix];
      nex[iz*_nx +       ix] *= tap[iz*_nx +       ix];
      cur[iz*_nx + _nx-ix-1] *= tap[iz*_nx + _nx-ix-1];
      nex[iz*_nx + _nx-ix-1] *= tap[iz*_nx + _nx-ix-1];
    }
  }
}

void scaas2d::get_info() {
  printf("\n Printing Propagation parameters\n");
  printf("nt=%d, dt=%f ot=0.0\n",_nt,_dt);
  printf("nz=%d, dz=%f oz=0.0\n",_nz,_dz);
  printf("nx=%d, dx=%f ox=0.0\n",_nx,_dx);
}
