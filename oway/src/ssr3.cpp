#include <math.h>
#include "ssr3.h"

ssr3::ssr3(int nx, int ny, int nz, int nh, int nw, int nr,
    float dx, float dy, float dz, float dh, float dw,
    float ow,float eps, int ntx, int nty) {
  /* Dimensions */
  _nx = nx; _ny = ny; _nz = nz; _nh = nh; _nw = nw; _nr = nr;
  _ntx = ntx; _nty = nty;
  _onestp = _nx*_ny;
  /* Samplings */
  _dx = dx; _dy = dy; _dz = dz; _dh = dh; _dw = dw;
  /* Frequency origin and stability */
  _ow = ow; _eps = eps;
  /* Initialize taper */
  build_taper(ntx,nty);
  //TODO: need to do all of the FFT configuration stuff here
}

void ssr3::ssr3ssf_modallw(float *slo, float *ref, std::complex<float> *wav, std::complex<float> *dat) {

  // wav dimensions
  // w is slowest
  // y is middle
  // x is fastest

  /* Loop over frequency (will be parallelized with OpenMP/TBB) */
  for(int iw = 0; iw < _nw; ++iw) {
    /* Get wavelet for current frequency */
    ssr3ssf_modonew(iw, slo, ref, wav + iw*_onestp, dat + iw*_onestp);
  }
}

void ssr3::ssr3ssf_modonew(int iw, float *slo, float *ref, std::complex<float> *wav, std::complex<float> *dat) {

  /* Allocate two temporary arrays */
  std::complex<float> *sslc = new std::complex<float>[_onestp]();
  std::complex<float> *rslc = new std::complex<float>[_onestp]();

  /* Taper the source wavefield (should it be done in place?) */

  /* Source loop over depth */
  for(int iz = 0; iz < _nz; ++iz) {
    /* Depth extrapolation */
  }

  /* Receiver loop over depth */
  for(int iz = 0; iz < _nz; ++iz) {
    /* Scattering with reflectivity */
    for(int iy = 0; iy < _ny; ++iy) {
      for(int ix = 0; ix < _nx; ++ix) {
      }
    }

    /* Depth extrapolation */
  }

  /* Taper the receiver wavefield */

  /* Free memory */
  delete[] sslc; delete[] rslc;
}

void ssr3::build_taper(int ntx, int nty) {
  if (ntx > 0) {
    _tapx = new float[ntx]();
    for (int it=0; it < ntx; it++) {
      float gain = sinf(0.5*M_PI*it/ntx);
      _tapx[it]=(1+gain)/2.;
    }
  }

  if (nty > 0) {
    _tapy = new float[nty]();
    for (int it=0; it < nty; it++) {
      float gain = sinf(0.5*M_PI*it/nty);
      _tapy[it]=(1+gain)/2.;
    }
  }
}

void ssr3::apply_taper(std::complex<float> *slcin, std::complex<float> *slcot) {
  int it,i2,i1;
  float gain;

//  for (int it=0; it < _nty; it++) {
//    float gain = _tapy[it];
//    for (int ix=0; ix < _nx; ix++) {
//     slcot[it      ][i1] = gain*slcin[it][it];
//     slcot[_ny-it-1][i1] = gain*slcin[_ny-it-1][i1];
//    }
//  }
//
//  for (it=0; it < tap->nt1; it++) {
//    gain = tap->tap1[it];
//    for (i2=0; i2 < tap->n2; i2++) {
//      if (tap->b1) tt[i2][        it  ] *= gain;
//      ;            tt[i2][tap->n1-it-1] *= gain;
//    }
//  }

}

void ssr3::apply_taper(std::complex<float> *slc) {
}
