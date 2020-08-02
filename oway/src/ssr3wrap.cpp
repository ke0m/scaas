#include <cstring>
#include "ssr3wrap.h"
#include "ssr3.h"
#include "progressbar.h"

void ssr3_modshots(int nx, int ny, int nz,
                   float ox, float oy, float oz,
                   float dx, float dy, float dz,
                   int nw, float ow, float dw,
                   int ntx, int nty, int px, int py,
                   float dtmax, int nrmax,
                   float *slo,
                   int nexp,
                   int *nsrc, float *srcy, float *srcx,
                   int *nrec, float *recy, float *recx,
                   std::complex<float>*wav,
                   float *ref,
                   std::complex<float>*dat,
                   int nthrds, int verb){
  /* Build SSR3 object */
  ssr3 ssf = ssr3(nx, ny, nz,
                  dx, dy, dz,
                  nw, ow, dw, 0.0,
                  ntx, nty, px, py,
                  dtmax, nrmax, nthrds);

  /* Set the slowness field */
  ssf.set_slows(slo);

  /* Allocate arrays */
  std::complex<float> *sou  = new std::complex<float>[nw*ny*nx]();
  std::complex<float> *datw = new std::complex<float>[nw*ny*nx]();

  /* Verbosity */
  bool everb = false, wverb = false;
  if(verb == 1) {
    everb = true;
  } else if(verb == 2) {
    wverb = true;
  }

  /* Loop over experiments */
  int ntr = 0, nwav = 0;
  for(int iexp = 0; iexp < nexp; ++iexp) {
    if(everb) printprogress("nexp", iexp, nexp);
    /* Initialize source and data arrays to zero */
    memset(sou ,0,sizeof(std::complex<float>)*nw*ny*nx);
    memset(datw,0,sizeof(std::complex<float>)*nw*ny*nx);
    /* Inject the sources */
    ssf.inject_srct(nsrc[iexp], srcy + nwav, srcx + nwav, oy, ox, wav + nwav*nw, sou);
    /* Downward continuation */
    ssf.ssr3ssf_modallw(ref,sou,datw,wverb);
    /* Restrict the data */
    ssf.restrict_datat(nrec[iexp], recy + ntr, recx + ntr, oy, ox, datw, dat + ntr*nw);
    /* Increment trace and wavelet positions */
    ntr += nrec[iexp]; nwav += nsrc[iexp];
  }

  /* Free memory */
  delete[] sou; delete[] datw;
}

void ssr3_migshots(int nx, int ny, int nz,
                   float ox, float oy, float oz,
                   float dx, float dy, float dz,
                   int nw, float ow, float dw,
                   int ntx, int nty, int px, int py,
                   float dtmax, int nrmax,
                   float *slo,
                   int nexp,
                   int *nsrc, float *srcy, float *srcx,
                   int *nrec, float *recy, float *recx,
                   std::complex<float>*dat,
                   std::complex<float>*wav,
                   float *img,
                   int nthrds, int verb) {
  /* Build SSR3 object */
  ssr3 ssf = ssr3(nx, ny, nz,
                  dx, dy, dz,
                  nw, ow, dw, 0.0,
                  ntx, nty, px, py,
                  dtmax, nrmax, nthrds);

  /* Set the slowness field */
  ssf.set_slows(slo);

  /* Allocate arrays */
  std::complex<float> *sou  = new std::complex<float>[nw*ny*nx]();
  std::complex<float> *datw = new std::complex<float>[nw*ny*nx]();
  float *iimg = new float[nz*ny*nx]();

  /* Verbosity */
  bool everb = false, wverb = false;
  if(verb == 1) {
    everb = true;
  } else if(verb == 2) {
    wverb = true;
  }

  /* Loop over experiments */
  int ntr = 0, nwav = 0;
  for(int iexp = 0; iexp < nexp; ++iexp) {
    if(everb) printprogress("nexp", iexp, nexp);
    /* Initialize source, data and image arrays to zero */
    memset(sou ,0,sizeof(std::complex<float>)*nw*ny*nx);
    memset(datw,0,sizeof(std::complex<float>)*nw*ny*nx);
    memset(iimg,0,sizeof(float)*nz*ny*nx);
    /* Inject the sources for this shot */
    ssf.inject_srct(nsrc[iexp], srcy + nwav, srcx + nwav, oy, ox, wav + nwav*nw, sou);
    /* Inject the data for this shot */
    ssf.inject_datat(nrec[iexp], recy + ntr, recx + ntr, oy, ox, dat + nw*ntr, datw);
    /* Imaging */
    ssf.ssr3ssf_migallw(datw,sou,iimg,wverb);
    /* Add to ouput image */
    for(int k = 0; k < nz*ny*nx; ++k) img[k] += iimg[k];
    /* Increment trace and wavelet positions */
    ntr += nrec[iexp]; nwav += nsrc[iexp];
  }

  /* Free memory */
  delete[] sou; delete[] datw; delete[] iimg;

}

void ssr3_migoffshots(int nx, int ny, int nz,
                      float ox, float oy, float oz,
                      float dx, float dy, float dz,
                      int nw, float ow, float dw,
                      int ntx, int nty, int px, int py,
                      float dtmax, int nrmax,
                      float *slo,
                      int nexp,
                      int *nsrc, float *srcy, float *srcx,
                      int *nrec, float *recy, float *recx,
                      std::complex<float>*dat,
                      std::complex<float>*wav,
                      int nhy, int nhx, bool sym,
                      float *img,
                      int nthrds, int verb){
  /* Build SSR3 object */
  ssr3 ssf = ssr3(nx, ny, nz,
                  dx, dy, dz,
                  nw, ow, dw, 0.0,
                  ntx, nty, px, py,
                  dtmax, nrmax, nthrds);

  /* Set the slowness field */
  ssf.set_slows(slo);

  /* Set extension */
  int rnhx, rnhy;
  if(sym) {
    rnhx = 2*nhx+1; rnhy = 2*nhy+1;
  } else {
    rnhx = nhx+1; rnhy = nhy+1;
  }
  ssf.set_ext(nhy, nhx, sym);


  /* Allocate arrays */
  std::complex<float> *sou  = new std::complex<float>[nw*ny*nx]();
  std::complex<float> *datw = new std::complex<float>[nw*ny*nx]();
  float *iimg = new float[rnhy*rnhx*nz*ny*nx]();

  /* Verbosity */
  bool everb = false, wverb = false;
  if(verb == 1) {
    everb = true;
  } else if(verb == 2) {
    wverb = true;
  }

  /* Loop over experiments */
  int ntr = 0, nwav = 0;
  for(int iexp = 0; iexp < nexp; ++iexp) {
    if(everb) printprogress("nexp", iexp, nexp);
    /* Initialize source, data and image arrays to zero */
    memset(sou ,0,sizeof(std::complex<float>)*nw*ny*nx);
    memset(datw,0,sizeof(std::complex<float>)*nw*ny*nx);
    memset(iimg,0,sizeof(float)*rnhy*rnhx*nz*ny*nx);
    /* Inject the sources for this shot */
    ssf.inject_srct(nsrc[iexp], srcy + nwav, srcx + nwav, oy, ox, wav + nwav*nw, sou);
    /* Inject the data for this shot */
    ssf.inject_datat(nrec[iexp], recy + ntr, recx + ntr, oy, ox, dat + nw*ntr, datw);
    /* Extended Imaging */
    ssf.ssr3ssf_migoffallw(datw, sou, iimg, wverb);
    /* Add to ouput image */
    for(int k = 0; k < rnhy*rnhx*nz*ny*nx; ++k) img[k] += iimg[k];
    /* Increment trace and wavelet positions */
    ntr += nrec[iexp]; nwav += nsrc[iexp];
  }

  /* Free extension memory */
  ssf.del_ext();

  /* Free memory */
  delete[] sou; delete[] datw; delete[] iimg;
}
