/*
 * fwi_funcs.cpp
 *
 *  Created on: Oct 5, 2018
 *      Author: joe
 */

#include <omp.h>
#include "laplacian10thispc.h"
#include "tapercos.h"
#include "fwi_funcs.h"

/**
 * Applies a delta function at the receiver locations on a wavefield slice
 * @param recz the receiver depth (assumes all receivers at same depth)
 * @param wslc the input wavefield slice
 * @param dslc the output slice of the data
 */
void drslc(int recz, float *wslc, float *dslc) {

  int nrx = dslc->get_axis(1).n; int drx = static_cast<int>(dslc->get_axis(1).d);
  int orx = static_cast<int>(dslc->get_axis(1).o);
  int nz = wslc->get_axis(1).n;

  /* Loop over all receivers for the time slice */
  for(int ir = 0; ir < nrx; ++ir) {
    int recloc = (ir)*drx + orx;
    dslc->vals[ir] = wslc->vals[recloc*nz + recz];
  }

}

void drtslc(int recz, hypercube_float *wslc, hypercube_float *dslc) {

  int nrx = dslc->get_axis(1).n; int drx = static_cast<int>(dslc->get_axis(1).d);
  int orx = static_cast<int>(dslc->get_axis(1).o);
  int nz = wslc->get_axis(1).n;

  for(int ir = 0; ir < nrx; ++ir) {
    int recloc = ir*(drx) + orx;
    wslc->vals[recloc*nz + recz] = dslc->vals[ir];
  }
}

void drslc_coords(int recz, hypercube_float *coords, hypercube_float *wslc, hypercube_float *dslc) {

  int nrx = coords->get_axis(1).n; int nz = wslc->get_axis(1).n;

  /* Loop over all receivers for the time slice */
  for(int ir = 0; ir < nrx; ++ir) {
    /* Nearest neighbor interpolation */
    int recloc = static_cast<int>(coords->vals[ir]);
    if(recloc >= 0) dslc->vals[ir] = wslc->vals[recloc*nz + recz];
  }
}


/** Applies a delta function at the receiver locations on a wavefield
 * @param recz The receiver depth
 * @param wfld the input wavefield
 * @param dat the output data
 */
void dr(int recz, hypercube_float *wfld, hypercube_float *dat) {
  /* Get axes */
  int nrx = dat->get_axis(1).n; int drx = static_cast<int>(dat->get_axis(1).d);
  int orx = static_cast<int>(dat->get_axis(1).o);

  int nz = wfld->get_axis(1).n; int nx = wfld->get_axis(2).n;
  int nt = wfld->get_axis(3).n;

  if(nt != dat->get_axis(2).n) {
    seperr("dr: wavefield and data must have same temporal axis.\n");
  }

  for(int it = 0; it < nt; ++it) {
    for(int ir = 0; ir < nrx; ++ir) {
      int recx = (ir)*drx + orx;
      dat->vals[it*nrx + ir] = wfld->vals[it*nx*nz + recx*nz + recz];
    }
  }
}

/**
 * Applies an injection (or interpolation) operator which is the adjoint
 * of the delta function
 * @param recz the receiver depth
 * @param wfld the output wavefield
 * @param dat the input data
 */
void drt(int recz, hypercube_float *wfld, hypercube_float *dat) {

  /* Get axes */
  int nrx = dat->get_axis(1).n; int drx = static_cast<int>(dat->get_axis(1).d);
  int orx = static_cast<int>(dat->get_axis(1).o);

  int nz = wfld->get_axis(1).n; int nx = wfld->get_axis(2).n;
  int nt = wfld->get_axis(3).n;

  if(nt != dat->get_axis(2).n) {
    seperr("drt: wavefield and data must have same temporal axis.\n");
  }

  for(int it = 0; it < nt; ++it) {
    for(int ir = 0; ir < nrx; ++ir) {
      int recx = (ir)*drx + orx;
      wfld->vals[it*nx*nz + recx*nz + recz] = dat->vals[it*nrx + ir];
    }
  }
}

void shot_interp(hypercube_float *datc, hypercube_float *datf) {
  /* Initialize */
  datf->zero();
  /* Get axes */
  int nrx = datc->get_axis(1).n; float orx = datc->get_axis(1).o; float drx = datc->get_axis(1).d;
  int ntc = datc->get_axis(2).n; float dtc = datc->get_axis(2).d;
  float dtf = datf->get_axis(2).d;
  if(datf->get_axis(1).n != nrx || datf->get_axis(1).o != orx || datf->get_axis(1).d != drx) {
    seperr("shot_interp: receiver axes must match.\n");
  }

  /* Allocate memory */
  std::vector<axis> raxis; raxis.push_back(datc->get_axis(1));
  hypercube_float *slc1 = new hypercube_float(raxis,true);
  hypercube_float *slc2 = new hypercube_float(raxis,true);
  hypercube_float *intp = new hypercube_float(raxis,true);
  slc1->zero(); slc2->zero(); intp->zero();

  /* Calculate coefficients */
  int sample = dtc/dtf;
  std::vector<float> ls1, ls2;
  for(int is = 0; is < sample; ++is) {
    ls1.push_back(static_cast<float>(sample-is)/static_cast<float>(sample));
    ls2.push_back(1.0-ls1[is]);
    ls1[is] /= sample; ls2[is] /= sample;
  }

  /* Interpolate from coarse to fine grid */
  int k = 0;
  for(int it = 0; it < ntc-1; ++it) {
    /* Grab values on coarse grid */
    memcpy(slc1->vals,&datc->vals[(it+0)*nrx],sizeof(float)*nrx);
    memcpy(slc2->vals,&datc->vals[(it+1)*nrx],sizeof(float)*nrx);
    for(int is = 0; is < sample; ++is) {
      /* Interpolate onto fine grid */
      intp->scale_add(1.0,slc1,ls1[is]);
      intp->scale_add(1.0,slc2,ls2[is]);
      /* Save to output */
      memcpy(&datf->vals[k*nrx],intp->vals,sizeof(float)*nrx);
      intp->zero();
      ++k;
    }
  }

  /* Free memory */
  delete slc1; delete slc2; delete intp;
}

void shot_interpt(hypercube_float *datc, hypercube_float *datf) {
  /* Initialize */
  datc->zero();
  /* Get axes */
  int nrx = datc->get_axis(1).n; float orx = datc->get_axis(1).o; float drx = datc->get_axis(1).d;
  int ntc = datc->get_axis(2).n; float dtc = datc->get_axis(2).d;
  float dtf = datf->get_axis(2).d;
  if(datf->get_axis(1).n != nrx || datf->get_axis(1).o != orx || datf->get_axis(1).d != drx) {
    seperr("shot_interpt: receiver axes must match.\n");
  }

  /* Allocate memory */
  std::vector<axis> raxis; raxis.push_back(datc->get_axis(1));
  hypercube_float *slc1 = new hypercube_float(raxis,true);
  hypercube_float *slc2 = new hypercube_float(raxis,true);
  hypercube_float *intp = new hypercube_float(raxis,true);
  slc1->zero(); slc2->zero(); intp->zero();

  /* Calculate coefficients */
  int sample = dtc/dtf;
  std::vector<float> ls1, ls2;
  for(int is = 0; is < sample; ++is) {
    ls1.push_back(static_cast<float>(sample-is)/static_cast<float>(sample));
    ls2.push_back(1.0-ls1[is]);
    ls1[is] /= sample; ls2[is] /= sample;
  }

  /* Interpolate from fine to coarse grid */
  int k = 0;
  for(int it = 0; it < ntc-1; ++it) {
    for(int is = 0; is < sample; ++is) {
      memcpy(intp->vals,&datf->vals[k*nrx],sizeof(float)*nrx);
      slc1->scale_add(1.0,intp,ls1[is]);
      slc2->scale_add(1.0,intp,ls2[is]);
      ++k;
    }
    /* Adjoint interpolate */
    memcpy(&datc->vals[(it+0)*nrx],slc1->vals,sizeof(float)*nrx);
    memcpy(&datc->vals[(it+1)*nrx],slc2->vals,sizeof(float)*nrx);
    /* Prepare for next time step */
    slc1->set(slc2->vals);
    slc2->zero();
  }

  /* Free memory */
  delete slc1; delete slc2; delete intp;

}

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
void fwdprop_wfld(hypercube_float *fin, hypercube_float *v,
    int bz, int bx, float alpha, hypercube_float *psol)  {

  /* Get axes */
  int nz = v->get_axis(1).n;
  int nx = v->get_axis(2).n;
  int nt = fin->get_axis(3).n; float dtu = fin->get_axis(3).d;

  if(nz != fin->get_axis(1).n || nx != fin->get_axis(2).n) {
    seperr("Velocity model and source spatial axes do not match.\n");
  }

  /* Useful values */
  int onestp = nx*nz;

  /* Precompute velocity dt^2 coefficient */
  hypercube_float *v2dt2 = (hypercube_float*)v->clone_vec();
  v2dt2->mult(v2dt2);
  v2dt2->scale(dtu * dtu);

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  int ord = 10;
  tapercos spg = tapercos(v->get_axes(),bz,bx,ord,alpha);

  /* Laplacian operator */
  laplacian10thispc lap = laplacian10thispc(fin->get_axes());

  /* Allocate memory for wavefield slices */
  hypercube_float *pre   = new hypercube_float(v->get_axes(),true);
  hypercube_float *cur   = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou1  = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou2  = new hypercube_float(v->get_axes(),true);
  hypercube_float *tmp;

  /* Initialize slices */
  pre->zero(); cur->zero();
  sou1->zero(); sou2->zero();

  /* Initial conditions */
  memcpy(psol->vals,pre->vals,sizeof(float)*onestp); //p(x,0) = 0.0

  memcpy(sou1->vals,&fin->vals[0*onestp],sizeof(float)*onestp);
  sou1->mult(v2dt2);
  spg.apply(sou1,pre);
  memcpy(&psol->vals[onestp],sou1->vals,sizeof(float)*onestp); // p(x,1) = v^2*dt^2*f(0)
  memcpy(cur->vals,sou1->vals,sizeof(float)*onestp);

  for(int it = 2; it < nt; ++it) {
    /* Calculate source term */
    memcpy(sou1->vals,&fin->vals[(it-1)*onestp],sizeof(float)*onestp);
    /* Apply laplacian */
    lap.forward(false, cur, sou2);
    /* Advance wavefields */
    for(int k = 0; k < onestp; ++k) {
      pre->vals[k] = v2dt2->vals[k]*sou1->vals[k] + 2*cur->vals[k] + v2dt2->vals[k]*sou2->vals[k] - pre->vals[k];
    }
    /* Apply taper */
    spg.apply(pre, cur);
    /* Copy to wavefield vector */
    memcpy(&psol->vals[it*onestp],pre->vals,sizeof(float)*onestp);
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Deallocate memory */
  delete v2dt2;
  delete pre; delete cur;
  delete sou1; delete sou2;

}

/**
 * Propagate the wavefield but using an input wavelet and save
 * the wavefield on the coarse time grid (data)
 */
void fwdprop_wfld_wavcrs(hypercube_float *wav, int srcx, int srcz, hypercube_float *v,
    int bz, int bx, float alpha, hypercube_float *psol)  {

  /* Get axes */
  int nz = v->get_axis(1).n;
  int nx = v->get_axis(2).n;
  int nt = wav->get_axis(1).n; float dtu = wav->get_axis(1).d;
  int ntd = psol->get_axis(3).n;
  int skip = nt/ntd;

  /* Useful values */
  int onestp = nx*nz;

  /* Precompute velocity dt^2 coefficient */
  hypercube_float *v2dt2 = (hypercube_float*)v->clone_vec();
  v2dt2->mult(v2dt2);
  v2dt2->scale(dtu * dtu);

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  int ord = 10;
  tapercos spg = tapercos(v->get_axes(),bz,bx,ord,alpha);

  /* Laplacian operator */
  laplacian10thispc lap = laplacian10thispc(v->get_axes());

  /* Allocate memory for wavefield slices */
  hypercube_float *pre   = new hypercube_float(v->get_axes(),true);
  hypercube_float *cur   = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou1  = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou2  = new hypercube_float(v->get_axes(),true);
  hypercube_float *tmp;

  /* Initialize slices */
  pre->zero(); cur->zero();
  sou1->zero(); sou2->zero();

  /* Initial conditions */
  memcpy(psol->vals,pre->vals,sizeof(float)*onestp); //p(x,0) = 0.0

  sou1->vals[srcx*nz + srcz] = wav->vals[0];
  sou1->mult(v2dt2);
  spg.apply(sou1,pre); // p(x,1) = v^2*dt^2*f(0)
  memcpy(cur->vals,sou1->vals,sizeof(float)*onestp);

  int k = 1;
  for(int it = 2; it < nt; ++it) {
    /* Calculate source term */
    sou1->vals[srcx*nz + srcz] = wav->vals[it-1];
    /* Apply laplacian */
    lap.forward(false, cur, sou2);
    /* Advance wavefields */
    for(int k = 0; k < onestp; ++k) {
      pre->vals[k] = v2dt2->vals[k]*sou1->vals[k] + 2*cur->vals[k] + v2dt2->vals[k]*sou2->vals[k] - pre->vals[k];
    }
    /* Apply taper */
    spg.apply(pre, cur);
    /* Copy to wavefield vector at coarse time step */
    if(it%skip == 0) {
      fprintf(stderr,"it=%d k=%d\n",it,k);
      memcpy(&psol->vals[k*onestp],pre->vals,sizeof(float)*onestp);
      k++;
    }
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Deallocate memory */
  delete v2dt2;
  delete pre; delete cur;
  delete sou1; delete sou2;

}

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
void fwdprop_data(hypercube_float *fin, hypercube_float *v,
    int bz, int bx, float alpha, int recz, hypercube_float *dat)  {

  /* Get axes */
  int nz = v->get_axis(1).n;
  int nx = v->get_axis(2).n;
  int nt = fin->get_axis(3).n; float dtu = fin->get_axis(3).d;
  std::vector<axis> raxis;
  raxis.push_back(dat->get_axis(1));
  int nrx = dat->get_axis(1).n;

  if(dat->get_axis(2).n != fin->get_axis(3).n) {
    seperr("fwdprop_data: Data must have time on slow (second axis)");
  }

  /* Useful values */
  int onestp = nx*nz;

  /* Precompute velocity dt^2 coefficient */
  hypercube_float *v2dt2 = (hypercube_float*)v->clone_vec();
  v2dt2->mult(v2dt2);
  v2dt2->scale(dtu * dtu);

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  int ord = 10;
  tapercos spg = tapercos(v->get_axes(),bz,bx,ord,alpha);

  /* Laplacian operator */
  laplacian10thispc lap = laplacian10thispc(fin->get_axes());

  /* Allocate memory for wavefield slices */
  hypercube_float *pre   = new hypercube_float(v->get_axes(),true);
  hypercube_float *cur   = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou1  = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou2  = new hypercube_float(v->get_axes(),true);
  hypercube_float *dslc  = new hypercube_float(raxis,true);
  hypercube_float *tmp;

  /* Initialize slices */
  pre->zero(); cur->zero();
  sou1->zero(); sou2->zero();
  dslc->zero();

  /* Initial conditions */
  memcpy(dat->vals,dslc->vals,sizeof(float)*nrx); //p(x,0) = 0.0

  memcpy(sou1->vals,&fin->vals[0*onestp],sizeof(float)*onestp);
  sou1->mult(v2dt2);
  spg.apply(sou1,pre);
  drslc(recz, sou1, dslc);
  memcpy(&dat->vals[nrx],dslc->vals,sizeof(float)*nrx); // p(x,1) = v^2*dt^2*f(0)
  memcpy(cur->vals,sou1->vals,sizeof(float)*onestp);

  for(int it = 2; it < nt; ++it) {
    /* Calculate source term */
    memcpy(sou1->vals,&fin->vals[(it-1)*onestp],sizeof(float)*onestp);
    /* Apply laplacian */
    lap.forward(false, cur, sou2);
    /* Advance wavefields */
    for(int k = 0; k < onestp; ++k) {
      pre->vals[k] = v2dt2->vals[k]*sou1->vals[k] + 2*cur->vals[k] + v2dt2->vals[k]*sou2->vals[k] - pre->vals[k];
    }
    /* Apply taper */
    spg.apply(pre, cur);
    /* Apply delta function operator */
    drslc(recz, pre, dslc);
    /* Copy to data vector */
    memcpy(&dat->vals[it*nrx],dslc->vals,sizeof(float)*nrx);
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Deallocate memory */
  delete v2dt2;
  delete pre; delete cur; delete dslc;
  delete sou1; delete sou2;

}

void fwdprop_data_wav(hypercube_float *wav, int srcz, int srcx, hypercube_float *v,
    int bz, int bx, float alpha, int recz, hypercube_float *dat)  {

  /* Get axes */
  int nz = v->get_axis(1).n;
  int nx = v->get_axis(2).n;
  int nt = wav->get_axis(1).n; float dtu = wav->get_axis(1).d;
  std::vector<axis> raxis;
  raxis.push_back(dat->get_axis(1));
  int nrx = dat->get_axis(1).n;

  if(dat->get_axis(2).n != wav->get_axis(1).n) {
    fprintf(stderr,"nt_data=%d nt_src=%d\n",dat->get_axis(2).n,wav->get_axis(1).n);
    seperr("fwdprop_data: Data must have time on slow (second axis)");
  }

  /* Useful values */
  int onestp = nx*nz;

  /* Precompute velocity dt^2 coefficient */
  hypercube_float *v2dt2 = (hypercube_float*)v->clone_vec();
  v2dt2->mult(v2dt2);
  v2dt2->scale(dtu * dtu);

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  int ord = 10;
  tapercos spg = tapercos(v->get_axes(),bz,bx,ord,alpha);

  /* Laplacian operator */
  laplacian10thispc lap = laplacian10thispc(v->get_axes());

  /* Allocate memory for wavefield slices */
  hypercube_float *pre   = new hypercube_float(v->get_axes(),true);
  hypercube_float *cur   = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou1  = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou2  = new hypercube_float(v->get_axes(),true);
  hypercube_float *dslc  = new hypercube_float(raxis,true);
  hypercube_float *tmp;

  /* Initialize slices */
  pre->zero(); cur->zero();
  sou1->zero(); sou2->zero();
  dslc->zero();

  /* Initial conditions */
  memcpy(dat->vals,dslc->vals,sizeof(float)*nrx); //p(x,0) = 0.0

  sou1->vals[srcx*nz + srcz] = wav->vals[0];
  sou1->mult(v2dt2);
  spg.apply(sou1,pre);
  drslc(recz, sou1, dslc);
  memcpy(&dat->vals[nrx],dslc->vals,sizeof(float)*nrx); // p(x,1) = v^2*dt^2*f(0)
  memcpy(cur->vals,sou1->vals,sizeof(float)*onestp);

  for(int it = 2; it < nt; ++it) {
    /* Calculate source term */
    sou1->vals[srcx*nz + srcz] = wav->vals[it-1];
    /* Apply laplacian */
    lap.forward(false, cur, sou2);
    /* Advance wavefields */
    for(int k = 0; k < onestp; ++k) {
      pre->vals[k] = v2dt2->vals[k]*sou1->vals[k] + 2*cur->vals[k] + v2dt2->vals[k]*sou2->vals[k] - pre->vals[k];
    }
    /* Apply taper */
    spg.apply(pre, cur);
    /* Apply delta function operator */
    drslc(recz, pre, dslc);
    /* Copy to data vector */
    memcpy(&dat->vals[it*nrx],dslc->vals,sizeof(float)*nrx);
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Deallocate memory */
  delete v2dt2;
  delete pre; delete cur; delete dslc;
  delete sou1; delete sou2;

}

void fwdprop_data_coords(hypercube_float *wav, int srcz, int srcx, hypercube_float *v,
    int bz, int bx, float alpha, int recz, hypercube_float *coords, hypercube_float *dat)  {

  /* Get axes */
  int nz = v->get_axis(1).n;
  int nx = v->get_axis(2).n;
  int nt = wav->get_axis(1).n; float dtu = wav->get_axis(1).d;
  std::vector<axis> raxis;
  raxis.push_back(dat->get_axis(1));
  int nrx = dat->get_axis(1).n;

  if(dat->get_axis(2).n != wav->get_axis(1).n) {
    fprintf(stderr,"nt_data=%d nt_src=%d\n",dat->get_axis(2).n,wav->get_axis(1).n);
    seperr("fwdprop_data: Data must have time on slow (second axis)");
  }


  /* Useful values */
  int onestp = nx*nz;

  /* Precompute velocity dt^2 coefficient */
  hypercube_float *v2dt2 = (hypercube_float*)v->clone_vec();
  v2dt2->mult(v2dt2);
  v2dt2->scale(dtu * dtu);

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  int ord = 10;
  tapercos spg = tapercos(v->get_axes(),bz,bx,ord,alpha);

  /* Laplacian operator */
  laplacian10thispc lap = laplacian10thispc(v->get_axes());

  /* Allocate memory for wavefield slices */
  hypercube_float *pre   = new hypercube_float(v->get_axes(),true);
  hypercube_float *cur   = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou1  = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou2  = new hypercube_float(v->get_axes(),true);
  hypercube_float *dslc  = new hypercube_float(raxis,true);
  hypercube_float *tmp;

  /* Initialize slices */
  pre->zero(); cur->zero();
  sou1->zero(); sou2->zero();
  dslc->zero();

  /* Initial conditions */
  memcpy(dat->vals,dslc->vals,sizeof(float)*nrx); //p(x,0) = 0.0

  sou1->vals[srcx*nz + srcz] = wav->vals[0];
  sou1->mult(v2dt2);
  spg.apply(sou1,pre);
  drslc_coords(recz, coords, sou1, dslc);
  memcpy(&dat->vals[nrx],dslc->vals,sizeof(float)*nrx); // p(x,1) = v^2*dt^2*f(0)
  memcpy(cur->vals,sou1->vals,sizeof(float)*onestp);

  for(int it = 2; it < nt; ++it) {
    /* Calculate source term */
    sou1->vals[srcx*nz + srcz] = wav->vals[it-1];
    /* Apply laplacian */
    lap.forward(false, cur, sou2);
    /* Advance wavefields */
    for(int k = 0; k < onestp; ++k) {
      pre->vals[k] = v2dt2->vals[k]*sou1->vals[k] + 2*cur->vals[k] + v2dt2->vals[k]*sou2->vals[k] - pre->vals[k];
    }
    /* Apply taper */
    spg.apply(pre, cur);
    /* Apply delta function operator */
    drslc_coords(recz, coords, pre, dslc);
    /* Copy to data vector */
    memcpy(&dat->vals[it*nrx],dslc->vals,sizeof(float)*nrx);
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Deallocate memory */
  delete v2dt2;
  delete pre; delete cur; delete dslc;
  delete sou1; delete sou2;

}

void adjprop(hypercube_float *ain, hypercube_float *v,
    int bz, int bx, float alpha, hypercube_float *lsol) {

  /* Get axes */
  int nz = v->get_axis(1).n;
  int nx = v->get_axis(2).n;
  int nt = ain->get_axis(3).n; float dtu = ain->get_axis(3).d;

  if(nz != ain->get_axis(1).n || nx != ain->get_axis(2).n) {
    seperr("Velocity model and adjoint source spatial axes do not match.\n");
  }

  /* Useful values */
  int onestp = nx*nz;

  /* Precompute velocity dt^2 coefficient */
  hypercube_float *v2dt2 = (hypercube_float*)v->clone_vec();
  v2dt2->mult(v2dt2);
  v2dt2->scale(dtu * dtu);

  /* Build taper for non-reflecting boundaries: assuming 10th order laplacian */
  int ord = 10;
  tapercos spg = tapercos(v->get_axes(),bz,bx,ord,alpha);

  /* Laplacian operator */
  laplacian10thispc lap = laplacian10thispc(ain->get_axes());

  /* Allocate memory for wavefield slices */
  hypercube_float *pre   = new hypercube_float(v->get_axes(),true);
  hypercube_float *cur   = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou1  = new hypercube_float(v->get_axes(),true);
  hypercube_float *sou2  = new hypercube_float(v->get_axes(),true);
  hypercube_float *tmp;

  /* Initialize slices */
  pre->zero(); cur->zero();
  sou1->zero(); sou2->zero();

  /* Terminal conditions */
  memcpy(&lsol->vals[onestp*(nt-1)],pre->vals,sizeof(float)*onestp); //l(x,nt-1) = 0.0

  memcpy(sou1->vals,&ain->vals[(nt-1)*onestp],sizeof(float)*onestp);
  sou1->mult(v2dt2); 
  spg.apply(sou1,pre);
  memcpy(&lsol->vals[onestp*(nt-2)],sou1->vals,sizeof(float)*onestp); // l(x,nt-2) = v^2*dt^2*asrc(nt-1)
  memcpy(cur->vals,sou1->vals,sizeof(float)*onestp);

  for(int it = nt-3; it > -1; --it) {
    /* Calculate source term */
    memcpy(sou1->vals,&ain->vals[(it+1)*onestp],sizeof(float)*onestp);
    /* Apply laplacian */
    lap.adjoint(false, sou2, cur);
    /* Advance wavefields */
    for(int k = 0; k < onestp; ++k) {
      pre->vals[k] = v2dt2->vals[k]*sou1->vals[k] + 2*cur->vals[k] + v2dt2->vals[k]*sou2->vals[k] - pre->vals[k];
    }
    /* Apply taper */
    spg.apply(pre, cur);
    /* Copy to wavefield vector */
    memcpy(&lsol->vals[it*onestp],pre->vals,sizeof(float)*onestp);
    /* Pointer swap */
    tmp = cur; cur = pre; pre = tmp;
  }

  /* Deallocate memory */
  delete v2dt2;
  delete pre; delete cur;
  delete sou1; delete sou2;
}

void d2t(hypercube_float* p ,hypercube_float *d2p) {
  /* Get time axis information */
  int nt = p->get_axis(3).n; float dt = p->get_axis(3).d;
  int nz = p->get_axis(1).n; int nx = p->get_axis(2).n;
  int onestp = (nx*nz);
  float idt2 = 1/(dt * dt);

  /* Initial rows */
  for(int k = 0; k < onestp; ++k) { d2p->vals[k] = p->vals[k]*idt2; }
  for(int k = 0; k < onestp; ++k) {
    d2p->vals[onestp + k] = (p->vals[onestp + k] - 2*p->vals[k])*idt2;
  }

  for(int it = 2; it < nt; ++it) {
    for(int k = 0; k < onestp; ++k) {
      d2p->vals[it*onestp + k] = (p->vals[it*onestp + k] - 2*p->vals[(it-1)*onestp + k] + p->vals[(it-2)*onestp + k])*idt2;
    }
  }
}

void calc_grad(hypercube_float *d2pt, hypercube_float *lsol, hypercube_float *v, hypercube_float* grad) {

  int nz = d2pt->get_axis(1).n; int nx = d2pt->get_axis(2).n; int nt = d2pt->get_axis(3).n;
  if(v->get_axis(1).n != nz || v->get_axis(2).n != nx) {
    seperr("calc_grad: wavefield and input velocity axes are different.\n");
  }

  int onestp = nx*nz;
  hypercube_float *ivel3 = new hypercube_float(v->get_axes(),true);
  ivel3->zero();

  /* Precompute velocity term */
  for(int k = 0; k < onestp; ++k) {
    if(v->vals[k] != 0.0) {
      ivel3->vals[k] = -2.0/(v->vals[k]*v->vals[k]*v->vals[k]);
    }
  }

  /* Gradient computation */
  for(int it = 1; it < nt; ++it) {
    for(int k = 0; k < onestp; ++k) {
      grad->vals[k] += ivel3->vals[k] * (lsol->vals[onestp*(it-1) + k] * d2pt->vals[onestp*it + k]);
    }
  }

  delete ivel3;

}

float gradient(hypercube_float *src, hypercube_float *vel, hypercube_float *dat, int recz, int srcz,
    int bz, int bx, float alpha, float z1, float z2, hypercube_float *grad, int nthd, bool *write_movies, invhelp *dgn) {

  int nz  = vel->get_axis(1).n; float dz = vel->get_axis(1).d;
  int nx  = vel->get_axis(2).n; float dx = vel->get_axis(2).d;
  int ntc = dat->get_axis(2).n;
  int nrx = dat->get_axis(1).n;
  int nsx = dat->get_axis(3).n; float osx = dat->get_axis(3).o; float dsx = dat->get_axis(3).d;
  /* Stencil padding */
  int pz = 5, px = 5, ord = 10;
  //int nzp = nz + 2*pz + ord; int nxp = nx + 2*px + ord;
  int nzp = nz + ord; int nxp = nx + ord;

  /* Padded velocity axis */
  std::vector<axis> vpaxes;
  vpaxes.push_back(axis(nzp,0.0,dz));
  vpaxes.push_back(axis(nxp,0.0,dx));

  /* Wavefield axes */
  std::vector<axis> paxes;
  paxes.push_back(vpaxes[0]);
  paxes.push_back(vpaxes[1]);
  paxes.push_back(src->get_axis(1));

  /* Axes for one shot */
  std::vector<axis> dfaxes, dcaxes;
  dcaxes.push_back(dat->get_axis(1));
  dcaxes.push_back(dat->get_axis(2));
  dfaxes.push_back(dat->get_axis(1));
  dfaxes.push_back(src->get_axis(1));

  /* Gradient container axes */
  std::vector<axis> gaxes;
  gaxes.push_back(vpaxes[0]);
  gaxes.push_back(vpaxes[1]);
  gaxes.push_back(dat->get_axis(3));

  /* Shot axis */
  std::vector<axis> shaxis;
  shaxis.push_back(dat->get_axis(3));

  /* Allocate memory */
  hypercube_float *velp   = new hypercube_float(vpaxes,true);
  hypercube_float *grdtmp = new hypercube_float(vpaxes,true);
  hypercube_float *grdtot = new hypercube_float(vpaxes,true);
  hypercube_float *grdcut = new hypercube_float(vel->get_axes(),true);
  hypercube_float *grds   = new hypercube_float(gaxes,true);
  hypercube_float *objs   = new hypercube_float(shaxis,true);
  hypercube_float *allrs  = new hypercube_float(dat->get_axes(),true);

  /* Initialize */
  velp->zero(); grdtmp->zero(); grdtot->zero(); grdcut->zero();
  grds->zero(); objs->zero(); allrs->zero(); grad->zero();

  /* Pad the velocity model */
  padvel pv = padvel(pz,px,ord);
  pv.forward(false, vel, velp);

  /* Loop over shots */
  omp_set_num_threads(nthd);
#pragma omp parallel for default(shared)
  for(int is = 0; is < nsx; ++is) {
    /* Allocate memory */
    hypercube_float *srcp  = new hypercube_float(paxes,true);
    hypercube_float *psol  = new hypercube_float(paxes,true);
    hypercube_float *d2p   = new hypercube_float(paxes,true);
    hypercube_float *lsol  = new hypercube_float(paxes,true);
    hypercube_float *dmod  = new hypercube_float(dfaxes,true);
    hypercube_float *res   = new hypercube_float(dcaxes,true);
    hypercube_float *obs   = new hypercube_float(dcaxes,true);
    hypercube_float *grd   = new hypercube_float(vpaxes,true);

    /* Initialize */
    srcp->zero(); psol->zero(); d2p->zero(); dmod->zero();
    lsol->zero(); res->zero(); obs->zero(); grd->zero();

    float srcx = osx + dsx*is;
    /* Pad the source wavelet */
    zeropad zp = zeropad(srcx, srcz);
    zp.forward(false, src, srcp);

    /* Forward propagation */
    fwdprop_wfld(srcp, velp, bz, bx, alpha, psol);

    /* Second derivative */
    d2t(psol,d2p);

    /* Delta function and interpolation operator */
    dr(recz,psol,dmod);
    shot_interpt(res,dmod);

    /* Calculate adjoint source */
    memcpy(obs->vals,&dat->vals[is*nrx*ntc],sizeof(float)*nrx*ntc);
    res->scale_add(-1.0, obs, 1.0);
    /* Interpolation and delta function adjoint */
    shot_interp(res, dmod);
    drt(recz,srcp,dmod);

    /* Adjoint propagation */
    adjprop(srcp, velp, bz, bx, alpha, lsol);

    /* Gradient calculation */
    calc_grad(d2p, lsol, velp, grd);
    memcpy(&grds->vals[is*nzp*nxp],grd->vals,sizeof(float)*nzp*nxp);

    /* Save residual */
    memcpy(&allrs->vals[is*nrx*ntc],res->vals,sizeof(float)*nrx*ntc);

    /* Save objective function value */
    objs->vals[is] = 0.5*res->dot(res);

    /* Free memory */
    delete srcp; delete psol; delete d2p; delete dmod;
    delete lsol; delete res; delete grd; delete obs;
  }

  /* Sum over all gradients */
  for(int ig = 0; ig < nsx; ++ig) {
    memcpy(grdtmp->vals,&grds->vals[ig*nzp*nxp],sizeof(float)*nzp*nxp);
    grdtot->scale_add(1.0, grdtmp, 1.0);
  }

  /* Undo padding and taper gradient */
  pv.adjoint(false, grdcut, grdtot);
  grad_taper tap = grad_taper(grdcut->get_axis(1),z1,z2);
  tap.apply(grdcut, grad);

  float objfxn = objs->sum();

  /* Write out files */
  if(*write_movies == true) {
    dgn->output_res(allrs);
    *write_movies = false;
  }

  /* Free memory */
  delete velp; delete grds; delete grdtmp; delete grdcut;
  delete grdtot; delete objs; delete allrs;

  return objfxn;

}

float indgradient(hypercube_float *src, hypercube_float *vel, hypercube_float *dat, int recz, int srcz,
    int bz, int bx, float alpha, float z1, float z2, std::vector<float> stimes, std::vector<int> xss,
    hypercube_float *grad, int nthd, bool *write_res, invhelp *dgn) {

  int nz  = vel->get_axis(1).n; float dz = vel->get_axis(1).d;
  int nx  = vel->get_axis(2).n; float dx = vel->get_axis(2).d;
  int ntc = dat->get_axis(2).n;
  int nrx = dat->get_axis(1).n;
  int nsx = (int)xss.size(); float osx = 0.0; float dsx = 1.0;
  /* Stencil padding */
  int pz = 5, px = 5, ord = 10;
  //int nzp = nz + 2*pz + ord; int nxp = nx + 2*px + ord;
  int nzp = nz + ord; int nxp = nx + ord;

  /* Padded velocity axis */
  std::vector<axis> vpaxes;
  vpaxes.push_back(axis(nzp,0.0,dz));
  vpaxes.push_back(axis(nxp,0.0,dx));

  /* Wavefield axes */
  std::vector<axis> paxes;
  paxes.push_back(vpaxes[0]);
  paxes.push_back(vpaxes[1]);
  paxes.push_back(src->get_axis(1));

  /* Axes for one shot */
  std::vector<axis> dcaxes, dfaxes;
  dcaxes.push_back(dat->get_axis(1));
  dcaxes.push_back(dat->get_axis(2));
  dfaxes.push_back(dat->get_axis(1));
  dfaxes.push_back(src->get_axis(1));

  /* Gradient container axes */
  std::vector<axis> gaxes;
  gaxes.push_back(vpaxes[0]);
  gaxes.push_back(vpaxes[1]);
  gaxes.push_back(dat->get_axis(3));

  /* Shot axis */
  std::vector<axis> shaxis;
  shaxis.push_back(axis(nsx,osx,dsx));

  /* Allocate memory */
  hypercube_float *velp   = new hypercube_float(vpaxes,true);
  hypercube_float *grdtmp = new hypercube_float(vpaxes,true);
  hypercube_float *grdtot = new hypercube_float(vpaxes,true);
  hypercube_float *grdcut = new hypercube_float(vel->get_axes(),true);
  hypercube_float *grds   = new hypercube_float(gaxes,true);
  hypercube_float *objs   = new hypercube_float(shaxis,true);
  hypercube_float *allrs  = new hypercube_float(dat->get_axes(),true);

  /* Initialize */
  velp->zero(); grdtmp->zero(); grdtot->zero(); grdcut->zero();
  grds->zero(); objs->zero(); allrs->zero(); grad->zero();

  /* Pad the velocity model */
  padvel pv = padvel(pz,px,ord);
  pv.forward(false, vel, velp);

  /* Loop over shots */
  omp_set_num_threads(nthd);
#pragma omp parallel for default(shared)
  for(int is = 0; is < nsx; ++is) {
    /* Allocate memory */
    hypercube_float *wshft = new hypercube_float(src->get_axes(),true);
    hypercube_float *srcp  = new hypercube_float(paxes,true);
    hypercube_float *psol  = new hypercube_float(paxes,true);
    hypercube_float *d2p   = new hypercube_float(paxes,true);
    hypercube_float *lsol  = new hypercube_float(paxes,true);
    hypercube_float *dmod  = new hypercube_float(dfaxes,true);
    hypercube_float *res   = new hypercube_float(dcaxes,true);
    hypercube_float *obs   = new hypercube_float(dcaxes,true);
    hypercube_float *grd   = new hypercube_float(vpaxes,true);

    /* Initialize */
    srcp->zero(); psol->zero(); d2p->zero(); dmod->zero();
    lsol->zero(); res->zero(); obs->zero(); grd->zero();

    /* Apply a shift to the wavelet */
    waveletpad wp = waveletpad(src->get_axis(1));
    wp.set_time(stimes[is]);
    wp.forward(false, src, wshft);
    /* Pad the source wavelet */
    float srcx = xss[is];

    /* Pad the source wavelet */
    zeropad zp = zeropad(srcx, srcz);
    zp.forward(false, wshft, srcp);

    /* Forward propagation */
    fwdprop_wfld(srcp, velp, bz, bx, alpha, psol);

    /* Second derivative */
    d2t(psol,d2p);

    /* Delta function operator */
    dr(recz,psol,dmod);
    shot_interpt(res, dmod);

    /* Calculate adjoint source */
    memcpy(obs->vals,&dat->vals[is*nrx*ntc],sizeof(float)*nrx*ntc);
    res->scale_add(-1.0, obs, 1.0);
    shot_interp(res,dmod);
    drt(recz,srcp,dmod);

    /* Adjoint propagation */
    adjprop(srcp, velp, bz, bx, alpha, lsol);

    /* Gradient calculation */
    calc_grad(d2p, lsol, velp, grd);
    memcpy(&grds->vals[is*nzp*nxp],grd->vals,sizeof(float)*nzp*nxp);

    /* Save residual */
    memcpy(&allrs->vals[is*nrx*ntc],res->vals,sizeof(float)*nrx*ntc);

    /* Save objective function value */
    objs->vals[is] = 0.5*res->dot(res);

    /* Free memory */
    delete srcp; delete psol; delete d2p; delete wshft;
    delete lsol; delete res; delete grd; delete obs;
    delete dmod;
  }

  /* Sum over all gradients */
  for(int ig = 0; ig < nsx; ++ig) {
    memcpy(grdtmp->vals,&grds->vals[ig*nzp*nxp],sizeof(float)*nzp*nxp);
    grdtot->scale_add(1.0, grdtmp, 1.0);
  }

  /* Undo padding and taper gradient */
  pv.adjoint(false, grdcut, grdtot);
  grad_taper tap = grad_taper(grdcut->get_axis(1),z1,z2);
  tap.apply(grdcut, grad);

  float objfxn = objs->sum();

  /* Write out files */
  if(*write_res == true) {
    dgn->output_res(allrs);
    *write_res = false;
  }

  /* Free memory */
  delete velp; delete grds; delete grdtmp; delete grdcut;
  delete grdtot; delete objs; delete allrs;

  return objfxn;

}

float bldgradient(hypercube_float *src, hypercube_float *vel, hypercube_float *dat, int recz, int srcz,
    int bz, int bx, float alpha, float z1, float z2, std::vector<std::vector<float>> stimes,
    std::vector<std::vector<int>> xss, std::vector<int> nb, hypercube_float *grad, int nthd, bool *write_res, invhelp *dgn) {

  int nz  = vel->get_axis(1).n; float dz = vel->get_axis(1).d;
  int nx  = vel->get_axis(2).n; float dx = vel->get_axis(2).d;
  int ntc = dat->get_axis(2).n;
  int nrx = dat->get_axis(1).n;
  int nsx = (int)nb.size(); float osx = 0.0; float dsx = 1.0;
  /* Stencil padding */
  int pz = 5, px = 5, ord = 10;
  //int nzp = nz + 2*pz + ord; int nxp = nx + 2*px + ord;
  int nzp = nz + ord; int nxp = nx + ord;

  /* Padded velocity axis */
  std::vector<axis> vpaxes;
  vpaxes.push_back(axis(nzp,0.0,dz));
  vpaxes.push_back(axis(nxp,0.0,dx));

  /* Wavefield axes */
  std::vector<axis> paxes;
  paxes.push_back(vpaxes[0]);
  paxes.push_back(vpaxes[1]);
  paxes.push_back(src->get_axis(1));

  /* Axes for one shot */
  std::vector<axis> dcaxes, dfaxes;
  dcaxes.push_back(dat->get_axis(1));
  dcaxes.push_back(dat->get_axis(2));
  dfaxes.push_back(dat->get_axis(1));
  dfaxes.push_back(src->get_axis(1));

  /* Gradient container axes */
  std::vector<axis> gaxes;
  gaxes.push_back(vpaxes[0]);
  gaxes.push_back(vpaxes[1]);
  gaxes.push_back(dat->get_axis(3));

  /* Shot axis */
  std::vector<axis> shaxis;
  shaxis.push_back(axis(nsx,osx,dsx));

  /* Allocate memory */
  hypercube_float *velp   = new hypercube_float(vpaxes,true);
  hypercube_float *grdtmp = new hypercube_float(vpaxes,true);
  hypercube_float *grdtot = new hypercube_float(vpaxes,true);
  hypercube_float *grdcut = new hypercube_float(vel->get_axes(),true);
  hypercube_float *grds   = new hypercube_float(gaxes,true);
  hypercube_float *objs   = new hypercube_float(shaxis,true);
  hypercube_float *allrs  = new hypercube_float(dat->get_axes(),true);

  /* Initialize */
  velp->zero(); grdtmp->zero(); grdtot->zero(); grdcut->zero();
  grds->zero(); objs->zero(); allrs->zero(); grad->zero();

  /* Pad the velocity model */
  padvel pv = padvel(pz,px,ord);
  pv.forward(false, vel, velp);

  /* Loop over shots */
  omp_set_num_threads(nthd);
#pragma omp parallel for default(shared)
  for(int is = 0; is < (int)nb.size(); ++is) {
    /* Allocate memory */
    hypercube_float *wshft = new hypercube_float(src->get_axes(),true);
    hypercube_float *srcp  = new hypercube_float(paxes,true);
    hypercube_float *psol  = new hypercube_float(paxes,true);
    hypercube_float *d2p   = new hypercube_float(paxes,true);
    hypercube_float *lsol  = new hypercube_float(paxes,true);
    hypercube_float *dmod  = new hypercube_float(dfaxes,true);
    hypercube_float *res   = new hypercube_float(dcaxes,true);
    hypercube_float *obs   = new hypercube_float(dcaxes,true);
    hypercube_float *grd   = new hypercube_float(vpaxes,true);

    /* Initialize */
    srcp->zero(); psol->zero(); d2p->zero(); dmod->zero();
    lsol->zero(); res->zero(); obs->zero(); grd->zero();

    waveletpad wp = waveletpad(src->get_axis(1));
    std::vector<int> xsi = xss[is];
    std::vector<float> stimej = stimes[is];

    /* Create blended source */
    for(int ib = 0; ib < nb[is]; ++ib) {
      /* Apply a time shift to wavelet */
      wp.set_time(stimej[ib]);
      wp.forward(false, src, wshft);
      /* Pad the source wavelet */
      zeropad zp = zeropad(xsi[ib], srcz);
      zp.forward(true, wshft, srcp);
    }

    /* Forward propagation */
    fwdprop_wfld(srcp, velp, bz, bx, alpha, psol);

    /* Second derivative */
    d2t(psol,d2p);

    /* Delta function and interpolation operator */
    dr(recz,psol,dmod);
    shot_interpt(res,dmod);

    /* Calculate adjoint source */
    memcpy(obs->vals,&dat->vals[is*nrx*ntc],sizeof(float)*nrx*ntc);
    res->scale_add(-1.0, obs, 1.0);
    /* Delta function and interpolation operator */
    shot_interp(res,dmod);
    drt(recz,srcp,dmod);

    /* Adjoint propagation */
    adjprop(srcp, velp, bz, bx, alpha, lsol);

    /* Gradient calculation */
    calc_grad(d2p, lsol, velp, grd);
    memcpy(&grds->vals[is*nzp*nxp],grd->vals,sizeof(float)*nzp*nxp);

    /* Save residual */
    memcpy(&allrs->vals[is*nrx*ntc],res->vals,sizeof(float)*nrx*ntc);

    /* Save objective function value */
    objs->vals[is] = 0.5*res->dot(res);

    /* Free memory */
    delete srcp; delete psol; delete d2p; delete wshft;
    delete lsol; delete res; delete grd; delete obs;
    delete dmod;
  }

  /* Sum over all gradients */
  for(int ig = 0; ig < nsx; ++ig) {
    memcpy(grdtmp->vals,&grds->vals[ig*nzp*nxp],sizeof(float)*nzp*nxp);
    grdtot->scale_add(1.0, grdtmp, 1.0);
  }

  /* Undo padding and taper gradient */
  pv.adjoint(false, grdcut, grdtot);
  grad_taper tap = grad_taper(grdcut->get_axis(1),z1,z2);
  tap.apply(grdcut, grad);

  float objfxn = objs->sum();

  /* Write out files */
  if(*write_res == true) {
    dgn->output_res(allrs);
    *write_res = false;
  }

  /* Free memory */
  delete velp; delete grds; delete grdtmp; delete grdcut;
  delete grdtot; delete objs; delete allrs;

  return objfxn;

}

void indmodel_data(hypercube_float *src, hypercube_float *vel, int recz, int srcz,
    int bz, int bx, float alpha, std::vector<float> stimes, std::vector<int> xss, int nthd, hypercube_float *idat) {

  int nz = vel->get_axis(1).n; float dz = vel->get_axis(1).d;
  int nx = vel->get_axis(2).n; float dx = vel->get_axis(2).d;
  int nrx = idat->get_axis(1).n; int ntc = idat->get_axis(2).n;
  int nsx = (int)xss.size();

  if(stimes.size() != xss.size()) {
    seperr("number of shot times %d should = number of shot positions %d\n",(int)stimes.size(),(int)xss.size());
  }
  /* Stencil padding */
  int pz = 5, px = 5, ord = 10;
  //int nzp = nz + 2*pz + ord; int nxp = nx + 2*px + ord;
  int nzp = nz + ord; int nxp = nx + ord;

  /* Padded velocity axis */
  std::vector<axis> vpaxes;
  vpaxes.push_back(axis(nzp,0.0,dz));
  vpaxes.push_back(axis(nxp,0.0,dx));

  /* Wavefield axes */
  std::vector<axis> paxes;
  paxes.push_back(vpaxes[0]);
  paxes.push_back(vpaxes[1]);
  paxes.push_back(src->get_axis(1));

  /* Single shot axes */
  std::vector<axis> raxes, riaxes;
  raxes.push_back(idat->get_axis(1)); riaxes.push_back(idat->get_axis(1));
  raxes.push_back(src->get_axis(1));  riaxes.push_back(idat->get_axis(2));

  /* Allocate memory */
  hypercube_float *velp   = new hypercube_float(vpaxes,true);

  /* Pad the velocity model */
  padvel pv = padvel(pz,px,ord);
  pv.forward(false, vel, velp);

  /* Loop over shots */
  omp_set_num_threads(nthd);
#pragma omp parallel for default(shared)
  for(int is = 0; is < nsx; ++is) {
    /* Temporary arrays */
    hypercube_float *wshft = new hypercube_float(src->get_axes(),true);
    hypercube_float *srcp  = new hypercube_float(paxes,true);
    hypercube_float *rect  = new hypercube_float(raxes,true);
    hypercube_float *recti = new hypercube_float(riaxes,true);
    /* Initialize */
    wshft->zero(); srcp->zero(); rect->zero(); recti->zero();
    /* Apply a shift to the wavelet */
    waveletpad wp = waveletpad(src->get_axis(1));
    wp.set_time(stimes[is]);
    wp.forward(false, src, wshft);
    /* Pad the source wavelet */
    float srcx = xss[is];
    zeropad zp = zeropad(srcx, srcz);
    zp.forward(false, wshft, srcp);
    /* Wave propagation */
    fwdprop_data(srcp, velp, bz, bx, alpha, recz, rect);
    /* Adjoint interpolation */
    shot_interpt(recti, rect);
    memcpy(&idat->vals[is*ntc*nrx],recti->vals,sizeof(float)*ntc*nrx);
    /* Free memory */
    delete srcp; delete rect; delete recti; delete wshft;
  }

  /* Free memory */
  delete velp;
}


void model_data(hypercube_float *src, hypercube_float *vel, int recz, int srcz,
    int bz, int bx, float alpha, int nthd, hypercube_float *dat) {

  /* Get input dimensions */
  int nz = vel->get_axis(1).n; float dz = vel->get_axis(1).d;
  int nx = vel->get_axis(2).n; float dx = vel->get_axis(2).d;
  int nrx = dat->get_axis(1).n; int ntc = dat->get_axis(2).n;
  int nsx = dat->get_axis(3).n; float osx = dat->get_axis(3).o; float dsx = dat->get_axis(3).d;

  /* Pad the velocity model */
  int pz = 5, px = 5, ord = 10;
  //int nzp = nz + 2*bz + ord; int nxp = nx + 2*bx + ord;
  int nzp = nz + ord; int nxp = nx + ord;

  /* Padded velocity axis */
  std::vector<axis> vpaxes;
  vpaxes.push_back(axis(nzp,0.0,dz));
  vpaxes.push_back(axis(nxp,0.0,dx));

  /* Wavefield axes */
  std::vector<axis> paxes;
  paxes.push_back(vpaxes[0]);
  paxes.push_back(vpaxes[1]);
  paxes.push_back(src->get_axis(1));

  /* Single shot axes */
  std::vector<axis> raxes, riaxes;
  raxes.push_back(dat->get_axis(1)); riaxes.push_back(dat->get_axis(1));
  raxes.push_back(src->get_axis(1));  riaxes.push_back(dat->get_axis(2));

  /* Allocate memory */
  hypercube_float *velp   = new hypercube_float(vpaxes,true);

  /* Pad the velocity model */
  padvel pv = padvel(pz+bz,px+bx,ord);
  pv.forward(false, vel, velp);

  /* Loop over shots */
  omp_set_num_threads(nthd);
#pragma omp parallel for default(shared)
  for(int is = 0; is < nsx; ++is) {
    /* Temporary arrays */
    hypercube_float *srcp = new hypercube_float(paxes,true);
    hypercube_float *rect = new hypercube_float(raxes,true);
    hypercube_float *recti = new hypercube_float(riaxes,true);
    /* Initialize arrays */
    srcp->zero(); rect->zero(); recti->zero();
    /* Pad the source wavelet */
    float srcx = osx + dsx*is;
    zeropad zp = zeropad(srcx, srcz);
    zp.forward(false, src, srcp);
    fwdprop_data(srcp, velp, bz, bx, alpha, recz, rect);
    shot_interpt(recti, rect);
    memcpy(&dat->vals[is*ntc*nrx],recti->vals,sizeof(float)*ntc*nrx);
    /* Free memory */
    delete srcp; delete rect; delete recti;
  }

}

void model_data_wav(hypercube_float *src, hypercube_float *vel, int recz, int srcz,
    int bz, int bx, float alpha, int nthd, hypercube_float *dat) {

  /* Get input dimensions */
  int nz = vel->get_axis(1).n; float dz = vel->get_axis(1).d;
  int nx = vel->get_axis(2).n; float dx = vel->get_axis(2).d;
  int nrx = dat->get_axis(1).n; int ntc = dat->get_axis(2).n;
  int nsx = dat->get_axis(3).n; float osx = dat->get_axis(3).o; float dsx = dat->get_axis(3).d;

  /* Pad the velocity model */
  int pz = 5, px = 5, ord = 10;
  //int nzp = nz + 2*bz + ord; int nxp = nx + 2*bx + ord;
  int nzp = nz + ord; int nxp = nx + ord;

  /* Padded velocity axis */
  std::vector<axis> vpaxes;
  vpaxes.push_back(axis(nzp,0.0,dz));
  vpaxes.push_back(axis(nxp,0.0,dx));

  /* Single shot axes */
  std::vector<axis> raxes, riaxes;
  raxes.push_back(dat->get_axis(1)); riaxes.push_back(dat->get_axis(1));
  raxes.push_back(src->get_axis(1));  riaxes.push_back(dat->get_axis(2));

  /* Receiver coordinate axes */
  std::vector<axis> rcaxes;
  rcaxes.push_back(axis(nrx,0.0,1.0));

  /* Allocate memory */
  hypercube_float *velp   = new hypercube_float(vpaxes,true);

  /* Pad the velocity model */
  padvel pv = padvel(pz+bz,px+bx,ord);
  pv.forward(false, vel, velp);

  /* Loop over shots */
  omp_set_num_threads(nthd);
#pragma omp parallel for default(shared)
  for(int is = 0; is < nsx; ++is) {
     /* Temporary arrays */
    hypercube_float *rect = new hypercube_float(raxes,true);
    hypercube_float *recti = new hypercube_float(riaxes,true);
    /* Initialize arrays */
    rect->zero(); recti->zero();
    /* Pad the source wavelet */
    float srcx = osx + dsx*is;
    fwdprop_data_wav(src,srcz,srcx,velp,bz,bx,alpha,recz,rect);
    shot_interpt(recti, rect);
    memcpy(&dat->vals[is*ntc*nrx],recti->vals,sizeof(float)*ntc*nrx);
    /* Free memory */
    delete rect; delete recti;
  }
}

void model_data_coords(hypercube_float *src, hypercube_float *vel, int recz,
    hypercube_float *reccoords, int srcz, hypercube_float *srccoords,
    int bz, int bx, float alpha, int nthd, hypercube_float *dat) {

  /* Get input dimensions */
  int nz = vel->get_axis(1).n; float dz = vel->get_axis(1).d;
  int nx = vel->get_axis(2).n; float dx = vel->get_axis(2).d;
  int nrx = dat->get_axis(1).n; int ntc = dat->get_axis(2).n;
  int nsx = dat->get_axis(3).n;

  if(nsx != srccoords->get_axis(1).n) {
    seperr("model_data_coords: must have same number of sources in data and coordinates.\n");
  }

  if(nrx != reccoords->get_axis(1).n) {
    seperr("model_data_coords: must have same number of receivers in data and coordinates.\n");
  }

  /* Pad the velocity model */
  int pz = 5, px = 5, ord = 10;
  //int nzp = nz + 2*bz + ord; int nxp = nx + 2*bx + ord;
  int nzp = nz + ord; int nxp = nx + ord;

  /* Padded velocity axis */
  std::vector<axis> vpaxes;
  vpaxes.push_back(axis(nzp,0.0,dz));
  vpaxes.push_back(axis(nxp,0.0,dx));

  /* Single shot axes */
  std::vector<axis> raxes, riaxes;
  raxes.push_back(dat->get_axis(1)); riaxes.push_back(dat->get_axis(1));
  raxes.push_back(src->get_axis(1));  riaxes.push_back(dat->get_axis(2));

  /* Receiver coordinate axes */
  std::vector<axis> rcaxes;
  rcaxes.push_back(axis(nrx,0.0,1.0));

  /* Allocate memory */
  hypercube_float *velp   = new hypercube_float(vpaxes,true);

  /* Pad the velocity model */
  padvel pv = padvel(pz+bz,px+bx,ord);
  pv.forward(false, vel, velp);

  /* Loop over shots */
  omp_set_num_threads(nthd);
#pragma omp parallel for default(shared)
  for(int is = 0; is < nsx; ++is) {
    /* Temporary arrays */
    hypercube_float *rect = new hypercube_float(raxes,true);
    hypercube_float *recti = new hypercube_float(riaxes,true);
    hypercube_float *coords = new hypercube_float(rcaxes,true);
    /* Initialize arrays */
    rect->zero(); recti->zero(); coords->zero();
    /* Get the source coordinate */
    int srcx = static_cast<int>(srccoords->vals[is]);
    /* Get the receiver coordinates for the current shot */
    memcpy(coords->vals,&reccoords->vals[is*nrx],sizeof(float)*nrx);
    fwdprop_data_coords(src, srcz, srcx, velp, bz, bx, alpha, recz, coords, rect);
    shot_interpt(recti, rect);
    memcpy(&dat->vals[is*ntc*nrx],recti->vals,sizeof(float)*ntc*nrx);
    /* Free memory */
    delete rect; delete recti; delete coords;
  }
}
