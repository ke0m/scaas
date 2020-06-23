#include <math.h>
#include <cstring>
#include "kiss_fft.h"
#include "ssr3.h"

ssr3::ssr3(int nx, int ny, int nz, int nh, int nw,
    float dx, float dy, float dz, float dh, float dw, float dtmax,
    float ow, float eps,
    int ntx, int nty, int px, int py, int nrmax, float *slo) {
  /* Dimensions */
  _nx  = nx;  _ny  = ny;  _nz = nz; _nh = nh; _nw = nw;
  _ntx = ntx; _nty = nty; _px = px; _py = py; _nrmax = nrmax;
  _bx = nx + px; _by = ny + py;
  _onestp = _nx*_ny;
  /* Samplings */
  _dx = dx; _dy = dy; _dz = dz; _dh = dh; _dw = dw;
  _dsmax = dtmax/_dz; _dsmax2 = _dsmax*_dsmax*_dsmax*_dsmax; // (slowness squared squared)
  /* Frequency origin and stability */
  _ow = ow; _eps = eps;
  /* Save a pointer to slowness */
  _slo = slo;
  /* Allocate reference slowness, taper, and wavenumber arrays */
  _sloref = new float[nrmax*nz](); _nr = new int[nz]();
  _tapx = new float[_ntx](); _tapy = new float[_nty]();
  _kk = new float[_bx*_by]();
  /* Build reference slownesses */
  build_refs(nz, _onestp, nrmax, _dsmax, slo, _nr, _sloref);
  /* Initialize taper */
  build_taper(ntx,nty,_tapx,_tapy);
  /* Build spatial frequencies */
  build_karray(_dx,_dy,_bx,_by,_kk);
  /* Forward and inverse FFTs */
  _fwd1 = kiss_fft_alloc(_bx,0,NULL,NULL);
  _inv1 = kiss_fft_alloc(_bx,1,NULL,NULL);
  _fwd2 = kiss_fft_alloc(_by,0,NULL,NULL);
  _inv2 = kiss_fft_alloc(_by,1,NULL,NULL);
}

void ssr3::ssr3ssf_modallw(float *ref, std::complex<float> *wav, std::complex<float> *dat) {

  // wav dimensions
  // w is slowest
  // y is middle
  // x is fastest

  /* Loop over frequency (will be parallelized with OpenMP/TBB) */
  for(int iw = 0; iw < _nw; ++iw) {
    /* Get wavelet for current frequency */
    ssr3ssf_modonew(iw, ref, wav + iw*_onestp, dat + iw*_onestp);
  }
}

void ssr3::ssr3ssf_modonew(int iw, float *ref, std::complex<float> *wav, std::complex<float> *dat) {

  /* Allocate two temporary arrays */
  std::complex<float> *sslc = new std::complex<float>[_nx*_ny*_nz]();
  std::complex<float> *rslc = new std::complex<float>[_onestp]();

  /* Current frequency */
  std::complex<float> w(_eps*_dw,_ow + iw*_dw);

  /* Taper the source wavefield */
  apply_taper(wav, sslc);

  /* Source loop over depth */
  for(int iz = 0; iz < _nz-1; ++iz) {
    /* Depth extrapolation */
    ssr3ssf(w, iz, _slo+(iz)*_onestp, _slo+(iz+1)*_onestp, sslc + iz*_ny*_nx);
  }

  /* Receiver loop over depth */
  for(int iz = _nz-1; iz > 0; --iz) {

    /* Scattering with reflectivity */
    for(int iy = 0; iy < _ny; ++iy) {
      for(int ix = 0; ix < _nx; ++ix) {
        sslc[iz*_ny*_nx + iy*_nx + ix] *= ref[iz*_ny*_nx + iy*_nx + ix];
        rslc[iy*_nx + ix] += sslc[iz*_ny*_nx + iy*_nx + ix];
      }
    }

    /* Depth extrapolation */
    ssr3ssf(w, iz, _slo+(iz)*_onestp, _slo+(iz-1)*_onestp, rslc);
  }

  /* Taper the receiver wavefield */
  apply_taper(rslc,dat);

  /* Free memory */
  delete[] sslc; delete[] rslc;
}

void ssr3::ssr3ssf(std::complex<float> w, int iz, float *scur, float *snex, std::complex<float> *wx) {

  /* Temporary arrays */
  std::complex<float> *pk  = new std::complex<float>[_bx*_by]();
  std::complex<float> *wk  = new std::complex<float>[_bx*_by]();
  float *wt = new float[_onestp]();

  std::complex<float> w2 = w*w;

  /* w-x-y part 1 */
  for(int iy = 0; iy < _ny; ++iy) {
    for(int ix = 0; ix < _nx; ++ix) {
      float s = 0.5 * scur[iy*_nx + ix];
      wx[iy*_nx + ix] *= exp(-w*s*_dz);
    }
  }

  /* FFT (w-x-y) -> (w-kx-ky) */
  memcpy(pk,wx,sizeof(std::complex<float>)*_onestp);
  fft2(false,(kiss_fft_cpx*)pk);

  memset(wx,0,sizeof(std::complex<float>)*(_bx*_by));

  /* Loop over reference velocities */
  for(int ir = 0; ir < _nr[iz]; ++ir) {

    /* w-kx-ky */
    std::complex<float> co = sqrt(w2 * _sloref[iz*_nrmax + ir]);
    for(int iky = 0; iky < _by; ++iky) {
      for(int ikx = 0; ikx < _bx; ++ikx) {
        std::complex<float> cc = sqrt(w2*_sloref[iz*_nrmax + ir] + _kk[iky*_bx + ikx]);
        wk[iky*_bx + ikx] = pk[iky*_bx + ikx] * exp((co-cc)*_dz);
      }
    }

    /* Inverse FFT (w-kx-ky) -> (w-x-y) */
    fft2(true,(kiss_fft_cpx*)wk);

    /* Interpolate (accumulate) */
    for(int iy = 0; iy < _ny; ++iy) {
      for(int ix = 0; ix < _nx; ++ix) {
        float d = fabsf(scur[iy*_nx + ix]*scur[iy*_nx + ix] - _sloref[iz*_nrmax + ir]);
        d = _dsmax2/(d*d + _dsmax2);
        wx[iy*_nx + ix] += wk[iy*_bx + ix]*d;
        wt[iy*_nx + ix] += d;
      }
    }
  }

  /* w-x part 2 */
  for(int iy = 0; iy < _ny; ++iy) {
    for(int ix = 0; ix < _nx; ++ix) {
      wx[iy*_nx + ix] /= wt[iy*_nx + ix];
      float s = 0.5 * snex[iy*_nx + ix];
      wx[iy*_nx + ix] *= exp(-w*s*_dz);
    }
  }

  apply_taper(wx);

  /* Free memory */
  delete[] pk; delete[] wk; delete[] wt;

}

void ssr3::build_refs(int nz, int ns, int nrmax, float ds, float *slo, int *nr, float *sloref) {

  for(int iz = 0; iz < nz; ++iz) {
    nr[iz] = nrefs(nrmax, ds, ns, slo + iz*ns, sloref + nrmax*iz);
  }
  for(int iz = 0; iz < nz-1; ++iz) {
    for(int ir = 0; ir < nr[iz]; ++ir) {
      sloref[iz*nrmax + ir] = 0.5*(sloref[iz*nrmax + ir] + sloref[(iz+1)*nrmax + ir]);
    }
  }

}

int ssr3::nrefs(int nrmax, float ds, int ns, float *slo, float *sloref) {

  /* Temporary slowness array */
  float *slo2 = new float[ns]();

  memcpy(slo2,slo,sizeof(float)*ns);

  float smax = quantile(ns-1,ns,slo2);
  float smin = quantile(   0,ns,slo2);
  nrmax = (nrmax < 1 + (smax-smin)/ds) ? (nrmax) : (1+(smax-smin)/ds);

  int jr = 0; float s2 = 0.0;
  for(int ir = 0; ir < nrmax; ++ir) {
    float qr = (ir + 1.0) - 0.5 * 1/nrmax;
    float s = quantile(qr*ns,ns,slo2);
    if(ir == 0 || fabsf(s-s2) > ds) {
      sloref[jr] = s*s;
      s2 = s;
      jr++;
    }
  }

  /* Free memory */
  delete[] slo2;

  return jr;

}

void ssr3::build_taper(int ntx, int nty,float *tapx, float *tapy) {
  if (ntx > 0) {
    for (int it=0; it < ntx; it++) {
      float gain = sinf(0.5*M_PI*it/ntx);
      tapx[it]=(1+gain)/2.;
    }
  }

  if (nty > 0) {
    for (int it=0; it < nty; it++) {
      float gain = sinf(0.5*M_PI*it/nty);
      tapy[it]=(1+gain)/2.;
    }
  }
}

void ssr3::apply_taper(std::complex<float> *slcin, std::complex<float> *slcot) {

  for (int it = 0; it < _nty; it++) {
    float gain = _tapy[it];
    for (int ix=0; ix < _nx; ix++) {
      slcot[(it      )*_nx + ix] = gain*slcin[(it      )*_nx + ix];
      slcot[(_ny-it-1)*_nx + ix] = gain*slcin[(_ny-it-1)*_nx + ix];
    }
  }

  for (int it = 0; it < _ntx; it++) {
    float gain = _tapx[it];
    for (int iy=0; iy < _ny; iy++) {
      slcot[iy*_nx +       it] = gain*slcin[iy*_nx +       it];
      slcot[iy*_nx + _nx-it-1] = gain*slcin[iy*_nx + _nx-it-1];
    }
  }

}

void ssr3::apply_taper(std::complex<float> *slc) {

  for (int it = 0; it < _nty; it++) {
    float gain = _tapy[it];
    for (int ix=0; ix < _nx; ix++) {
      slc[(it      )*_nx + ix] *= gain;
      slc[(_ny-it-1)*_nx + ix] *= gain;
    }
  }

  for (int it = 0; it < _ntx; it++) {
    float gain = _tapx[it];
    for (int iy=0; iy < _ny; iy++) {
      slc[iy*_nx +       it] *= gain;
      slc[iy*_nx + _nx-it-1] *= gain;
    }
  }
}

void ssr3::build_karray(float dx, float dy, int bx, int by, float *kk) {

  /* Spatial frequency axes */
  float dkx = 2.0*M_PI/(bx*dx); float okx = (bx == 1) ? 0 : -M_PI/dx;
  float dky = 2.0*M_PI/(by*dy); float oky = (bx == 1) ? 0 : -M_PI/dy;

  /* Populate the array */
  for(int iy = 0; iy < by; ++iy) {
    int jy = (iy < by/2.0) ? (iy + by/2.0) : (iy-by/2.0);
    float ky = oky + jy*dky;
    for(int ix = 0; ix < bx; ++ix) {
      int jx = (ix < bx/2.0) ? (ix + bx/2.0) : (ix-bx/2.0);
      float kx = okx + jx*dkx;
      kk[iy*bx + ix] = kx*kx + ky*ky;
    }
  }
}

void ssr3::fft2(bool inv, kiss_fft_cpx *pp) {

  kiss_fft_cpx *ctrace = new kiss_fft_cpx[_by];

  if(inv) {
    /* IFT 1 */
    for(int iy = 0; iy < _by; ++iy) {
      kiss_fft(_inv1, pp + _bx*iy, pp + _bx*iy);
    }

    /* IFT 2 */
    for(int ix = 0; ix < _bx; ++ix) {
      kiss_fft_stride(_inv2, pp+ix, ctrace, _bx);
      for(int iy = 0; iy < _by; ++iy) {
        pp[iy*_bx + ix] = ctrace[iy];
      }
    }

    /* Scaling */
    for(int iy = 0; iy < _by; ++iy) {
      for(int ix = 0; ix < _bx; ++ix) {
        pp[iy*_bx + ix] = cmul(pp[iy*_bx + ix],1/(sqrtf(_bx*_by)));
      }
    }

  } else {

    /* Scaling */
    for(int iy = 0; iy < _by; ++iy) {
      for(int ix = 0; ix < _bx; ++ix) {
        pp[iy*_bx + ix] = cmul(pp[iy*_bx + ix],1/sqrtf(_bx*_by));
      }
    }

    /* FFT 2 */
    for(int ix = 0; ix < _bx; ++ix) {
      kiss_fft_stride(_fwd2,pp + ix, ctrace, _bx);
      for(int iy = 0; iy < _by; ++iy) {
        pp[iy*_bx + ix] = ctrace[iy];
      }
    }

    /* FFT 1 */
    for(int iy = 0; iy < _ny; ++iy) {
      kiss_fft(_fwd1, pp + iy*_bx, pp + iy*_bx);
    }

  }

  delete[] ctrace;
}

kiss_fft_cpx ssr3::cmul(kiss_fft_cpx a, float b) {
  kiss_fft_cpx c;

  c.r = a.r/b; c.i = a.i/b;
  return c;

}

float ssr3::quantile(int q, int n, float *a) {
  float *low = a; float *hi  = a+n-1; float *k=a+q;
  while (low<hi) {
    float ak = *k;
    float *i = low; float *j = hi;
    do {
      while (*i < ak) i++;
      while (*j > ak) j--;
      if (i<=j) {
        float buf = *i;
        *i++ = *j;
        *j-- = buf;
      }
    } while (i<=j);
    if (j<k) low = i;
    if (k<i) hi = j;
  }
  return (*k);
}

