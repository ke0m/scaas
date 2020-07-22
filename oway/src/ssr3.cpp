#include <math.h>
#include <cstring>
#include <omp.h>
#include "kiss_fft.h"
#include "ssr3.h"
#include "progressbar.h"
#include <map>
#include <string>
#include "/opt/matplotlib-cpp/matplotlibcpp.h"

namespace plt = matplotlibcpp;

void plotimg_cmplx(int n1,int n2,std::complex<float> *arr,int option) {
  float * tmp = new float[n1*n2];

  for(int i1 = 0; i1 < n1; ++i1) {
    for(int i2 = 0; i2 < n2; ++i2) {
      if(option == 0) {
        tmp[i1*n2 + i2] = std::real(arr[i1*n2 + i2]);
      } else if(option == 1) {
        tmp[i1*n2 + i2] = std::imag(arr[i1*n2 + i2]);
      } else {
        tmp[i1*n2 + i2] = std::abs(arr[i1*n2 + i2]);
      }
    }
  }
  std::map<std::string,std::string> vals;
  //vals["vmax"] = "0.01"; //vals["vmin"] = "0.0";
  plt::imshow((const float *)tmp,n1,n2,1,vals); plt::show();

  delete [] tmp;
}

void plotplt_cmplx(int n1, std::complex<float> *arr, int option) {
  float * tmp = new float[n1];

  for(int i1 = 0; i1 < n1; ++i1) {
    if(option == 0) {
      tmp[i1] = std::real(arr[i1]);
    } else if(option == 1) {
      tmp[i1] = std::imag(arr[i1]);
    } else {
      tmp[i1] = std::abs(arr[i1]);
    }
  }
  std::vector<float> v {tmp,tmp+n1};
  plt::plot(v); plt::show();

  delete [] tmp;

}

ssr3::ssr3(int nx,   int ny,   int nz,
    float dx, float dy, float dz,
    int nw,   float ow, float dw, float eps,
    int ntx, int nty, int px, int py,
    float dtmax, int nrmax) {
  /* Dimensions */
  _nx  = nx;  _ny  = ny;  _nz = nz; _nw = nw;
  _ntx = ntx; _nty = nty; _px = px; _py = py; _nrmax = nrmax;
  _bx = nx + px; _by = ny + py;
  /* Samplings */
  _dx = dx; _dy = dy; _dz = dz; _dw = 2*M_PI*dw;
  _dsmax = dtmax/_dz; _dsmax2 = _dsmax*_dsmax*_dsmax*_dsmax; // (slowness squared squared)
  /* Frequency origin and stability */
  _ow = 2*M_PI*ow; _eps = eps;
  /* Allocate reference slowness, taper, and wavenumber arrays */
  _sloref = new float[nrmax*nz](); _nr = new int[nz]();
  _tapx = new float[_ntx](); _tapy = new float[_nty]();
  _kk = new float[_bx*_by]();
  /* Initialize taper */
  build_taper(ntx,nty,_tapx,_tapy);
  /* Build spatial frequencies */
  build_karray(_dx,_dy,_bx,_by,_kk);
  /* Forward and inverse FFTs */
  _fwd1 = kiss_fft_alloc(_bx,0,NULL,NULL);
  _inv1 = kiss_fft_alloc(_bx,1,NULL,NULL);
  _fwd2 = kiss_fft_alloc(_by,0,NULL,NULL);
  _inv2 = kiss_fft_alloc(_by,1,NULL,NULL);
  /* Save a pointer to slowness */
  _slo = NULL;
}

void ssr3::set_slows(float *slo) {

  /* First set the migration slowness */
  _slo = slo;

  /* Compute number of reference slownesses with depth */
  for(int iz = 0; iz < _nz; ++iz) {
    _nr[iz] = nrefs(_nrmax, _dsmax, _nx*_ny, _slo + iz*_nx*_ny, _sloref + _nrmax*iz);
  }

  /* Build reference slownesses */
  for(int iz = 0; iz < _nz-1; ++iz) {
    for(int ir = 0; ir < _nr[iz]; ++ir) {
      _sloref[iz*_nrmax + ir] = 0.5*(_sloref[iz*_nrmax + ir] + _sloref[(iz+1)*_nrmax + ir]);
    }
  }

}

void ssr3::ssr3ssf_modallw(float *ref, std::complex<float> *wav, std::complex<float> *dat, int nthrds, bool verb) {

  /* Check if built reference velocities */
  if(_slo == NULL) {
    fprintf(stderr,"Must run set_slows before modeling or migration\n");
  }

  /* Set up printing if verbosity is desired */
  int *widx = new int[nthrds]();
  int csize = (int)_nw/nthrds;
  bool firstiter = true;

  /* Loop over frequency */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iw = 0; iw < _nw; ++iw) {
    /* Verbosity */
    int wthd = omp_get_thread_num();
    if(firstiter && verb) widx[wthd] = iw;
    if(verb) printprogress_omp("nw:",iw-widx[wthd],csize,wthd);
    /* Get wavelet and model data for current frequency */
    ssr3ssf_modonew(iw, ref, wav + iw*_nx*_ny, dat + iw*_nx*_ny);
    firstiter = false;
  }
  if(verb) printf("\n");

  delete[] widx;
}

void ssr3::ssr3ssf_modonew(int iw, float *ref, std::complex<float> *wav, std::complex<float> *dat) {

  /* Allocate two temporary arrays */
  std::complex<float> *sslc = new std::complex<float>[_ny*_nx*_nz](); // (z,y,x)
  std::complex<float> *rslc = new std::complex<float>[_ny*_nx]();     //   (y,x)

  /* Current frequency */
  std::complex<float> w(_eps*_dw,_ow + iw*_dw);

  /* Taper the source wavefield */
  apply_taper(wav, sslc);

  /* Source loop over depth */
  for(int iz = 0; iz < _nz-1; ++iz) {
    /* Depth extrapolation */
    ssr3ssf(w, iz, _slo+(iz)*_nx*_ny, _slo+(iz+1)*_nx*_ny, sslc + (iz)*_nx*_ny, sslc + (iz+1)*_nx*_ny);
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
    ssr3ssf(w, iz, _slo+(iz)*_nx*_ny, _slo+(iz-1)*_nx*_ny, rslc);
  }

  /* Taper the receiver wavefield */
  apply_taper(rslc,dat);

  /* Free memory */
  delete[] sslc; delete[] rslc;
}

void ssr3::restrict_data(int nrec, float *recy, float *recx, float oy, float ox,
                         std::complex<float> *dat, std::complex<float> *rec) {

  /* Loop over receivers */
  for(int ir = 0; ir < nrec; ++ir) {
    int iry = (recy[ir]-oy)/_dy + 0.5;
    int irx = (recx[ir]-ox)/_dx + 0.5;
    if(iry >= 0 && iry < _ny && irx >=0 && irx < _nx) {
      memcpy(&rec[ir*_nw],&dat[iry*_nw*_nx + irx*_nw],sizeof(std::complex<float>)*_nw);
    }
  }

}

void ssr3::inject_data(int nrec, float *recy, float *recx, float oy, float ox,
                       std::complex<float> *rec, std::complex<float> *dat) {

  /* Loop over receivers */
  for(int ir = 0; ir < nrec; ++ir) {
    int iry = (recy[ir]-oy)/_dy + 0.5;
    int irx = (recx[ir]-ox)/_dx + 0.5;
    if(iry >= 0 && iry < _ny && irx >= 0 && irx < _nx) {
      memcpy(&dat[iry*_nw*_nx + irx*_nw],&rec[ir*_nw],sizeof(std::complex<float>)*_nw);
    }
  }
}

void ssr3::ssr3ssf_migallw(std::complex<float> *dat, std::complex<float> *wav, float *img, int nthrds, bool verb) {
  /* Check if built reference velocities */
  if(_slo == NULL) {
    fprintf(stderr,"Must run set_slows before modeling or migration\n");
  }

  /* Set up printing if verbosity is desired */
  int *widx = new int[nthrds]();
  int csize = (int)_nw/nthrds;
  bool firstiter = true;

  /* Loop over frequency */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iw = 0; iw < _nw; ++iw) {
    /* Verbosity */
    int wthd = omp_get_thread_num();
    if(firstiter && verb) widx[wthd] = iw;
    if(verb) printprogress_omp("nw:",iw-widx[wthd],csize,wthd);
    /* Migrate data for current frequency */
    ssr3ssf_migonew(iw, dat + iw*_nx*_ny, wav + iw*_nx*_ny, img);
    firstiter = false;
  }
  if(verb) printf("\n");

  delete[] widx;
}

void ssr3::ssr3ssf_migonew(int iw, std::complex<float> *dat, std::complex<float> *wav, float *img) {

  /* Temporary arrays (depth slices) */
  std::complex<float> *sslc = new std::complex<float>[_ny*_nx]();
  std::complex<float> *rslc = new std::complex<float>[_ny*_nx]();

  /* Current frequency */
  std::complex<float> ws(_eps*_dw,+(_ow + iw*_dw)); // Causal
  std::complex<float> wr(_eps*_dw,-(_ow + iw*_dw)); // Anti-causal

  apply_taper(wav, sslc);
  apply_taper(dat, rslc);

  /* Loop over all depths (note goes up to nz-1) */
  for(int iz = 0; iz < _nz-1; ++iz) {
    /* Depth extrapolation of source and receiver wavefields */
    ssr3ssf(ws, iz, _slo+(iz)*_nx*_ny, _slo+(iz+1)*_nx*_ny, sslc);
    ssr3ssf(wr, iz, _slo+(iz)*_nx*_ny, _slo+(iz+1)*_nx*_ny, rslc);
    /* Imaging condition */
    for(int iy = 0; iy < _ny; ++iy) {
      for(int ix = 0; ix < _nx; ++ix) {
        img[iz*_nx*_ny + iy*_nx + ix] += std::real(std::conj(sslc[iy*_nx + ix])*rslc[iy*_nx + ix]);
      }
    }
  }
  /* Free memory */
  delete[] sslc; delete[] rslc;
}

void ssr3::ssr3ssf_migoffallw(std::complex<float> *dat, std::complex<float> *wav, int nhy, int nhx, bool sym, float *img,
                              int nthrds, bool verb) {
  /* Check if built reference velocities */
  if(_slo == NULL) {
    fprintf(stderr,"Must run set_slows before modeling or migration\n");
  }

  /* Compute the bounds for the lags in the imaging condition */
  int blx, elx, bly, ely;
  if(sym) {
    blx = -nhx; elx = nhx;
    bly = -nhy; ely = nhy;
  } else {
    blx = 0; elx = nhx;
    bly = 0; ely = nhy;
  }

  /* Set up printing if verbosity is desired */
  int *widx = new int[nthrds]();
  int csize = (int)_nw/nthrds;
  bool firstiter = true;

  /* Loop over frequency */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iw = 0; iw < _nw; ++iw) {
    /* Verbosity */
    int wthd = omp_get_thread_num();
    if(firstiter && verb) widx[wthd] = iw;
    if(verb) printprogress_omp("nw:",iw-widx[wthd],csize,wthd);
    /* Migrate data for current frequency */
    ssr3ssf_migoffonew(iw, dat + iw*_nx*_ny, wav + iw*_nx*_ny, bly, ely, blx, elx, img);
    firstiter = false;
  }
  if(verb) printf("\n");

  delete[] widx;
}

void ssr3::ssr3ssf_migoffonew(int iw, std::complex<float> *dat, std::complex<float>*wav,
                              int bly, int ely, int blx, int elx, float *img) {
  /* Temporary arrays (depth slices) */
  std::complex<float> *sslc = new std::complex<float>[_ny*_nx]();
  std::complex<float> *rslc = new std::complex<float>[_ny*_nx]();

  /* Current frequency */
  std::complex<float> ws(_eps*_dw,+(_ow + iw*_dw)); // Causal
  std::complex<float> wr(_eps*_dw,-(_ow + iw*_dw)); // Anti-causal

  /* Sizes and shifts */
  int begx = elx; int endx = _nx - begx;
  int begy = ely; int endy = _ny - begy;
  int nhx  = elx - blx;
  int shfx = abs(blx); int shfy = abs(bly);

  apply_taper(wav, sslc);
  apply_taper(dat, rslc);

  /* Loop over all depths */
  for(int iz = 0; iz < _nz-1; ++iz) {
    /* Depth extrapolation of source and receiver wavefields */
    ssr3ssf(ws, iz, _slo+(iz)*_nx*_ny, _slo+(iz+1)*_nx*_ny, sslc);
    ssr3ssf(wr, iz, _slo+(iz)*_nx*_ny, _slo+(iz+1)*_nx*_ny, rslc);
    /* Loops over lags */
#pragma omp critical
    for(int ily = bly; ily <= ely; ++ily) {
      for(int ilx = blx; ilx <= elx; ++ilx) {
        /* Imaging condition (should do this on ISPC) */
        for(int iy = begy; iy < endy; ++iy) {
          for(int ix = begx; ix < endx; ++ix) {
            int imgidx = iz*_nx*_ny + iy*_nx + ix;
            img[(ily+shfy)*_nx*_ny*_nz*nhx + (ilx+shfx)*_nx*_ny*_nz + imgidx]  +=
                                 std::real(std::conj(sslc[(iy-ily)*_nx + (ix-ilx)])*rslc[(iy+ily)*_nx + (ix+ilx)]);
          } // x
        } // y
      } // lx
    } // ly
  } // z
  /* Free memory */
  delete[] sslc; delete[] rslc;
}

void ssr3::ssr3ssf_modallwzo(float *img, std::complex<float> *dat, int nthrds, bool verb) {
  /* Check if built reference velocities */
  if(_slo == NULL) {
    fprintf(stderr,"Must run set_slows before modeling or migration\n");
  }

  /* Set up printing if verbosity is desired */
  int *widx = new int[nthrds]();
  int csize = (int)_nw/nthrds;
  bool firstiter = true;

  /* Loop over frequency */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iw = 0; iw < _nw; ++iw) {
    /* Verbosity */
    int wthd = omp_get_thread_num();
    if(firstiter && verb) widx[wthd] = iw;
    if(verb) printprogress_omp("nw:",iw-widx[wthd],csize,wthd);
    /* Get wavelet and model data for current frequency */
    ssr3ssf_modonewzo(iw, img, dat + iw*_nx*_ny);
    firstiter = false;
  }
  if(verb) printf("\n");

  delete[] widx;
}

void ssr3::ssr3ssf_modonewzo(int iw, float *img, std::complex<float> *dat) {

  /* Current frequency */
  std::complex<float> w(_eps*_dw,+(_ow + iw*_dw)); // Causal

  /* Loop over depth */
  for(int iz = _nz-1; iz > 0; --iz) {
    /* Boundary (imaging) condition */
    for(int iy = 0; iy < _ny; ++iy) {
      for(int ix = 0; ix < _nx; ++ix) {
        dat[iy*_nx + ix] += std::complex<float>(img[iz*_ny*_nx + iy*_nx + ix]);
      }
    }
    /* Upward continuation */
    ssr3ssf(w, iz, _slo+(iz)*_nx*_ny, _slo+(iz-1)*_nx*_ny, dat);
  }
  /* Data=Image at z=0 */
  for(int iy = 0; iy < _ny; ++iy) {
    for(int ix = 0; ix < _nx; ++ix) {
      dat[iy*_nx + ix] += std::complex<float>(img[_ny*_nx + iy*_nx + ix],0.0);
    }
  }

  /* Apply taper to output */
  apply_taper(dat);
}

void ssr3::ssr3ssf_migallwzo(std::complex<float> *dat, float *img, int nthrds, bool verb) {

  /* Check if built reference velocities */
  if(_slo == NULL) {
    fprintf(stderr,"Must run set_slows before modeling or migration\n");
  }

  /* Set up printing if verbosity is desired */
  int *widx = new int[nthrds]();
  int csize = (int)_nw/nthrds;
  bool firstiter = true;

  /* Loop over frequency */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int iw = 0; iw < _nw; ++iw) {
    /* Verbosity */
    int wthd = omp_get_thread_num();
    if(firstiter && verb) widx[wthd] = iw;
    if(verb) printprogress_omp("nw:",iw-widx[wthd],csize,wthd);
    /* Migrate data for current frequency */
    ssr3ssf_migonewzo(iw, dat + iw*_nx*_ny, img);
    firstiter = false;
  }
  if(verb) printf("\n");

  delete[] widx;
}

void ssr3::ssr3ssf_migonewzo(int iw, std::complex<float> *dat, float *img) {
  /* Temporary array (depth slice) */
  std::complex<float> *slc = new std::complex<float>[_ny*_nx]();

  /* Current frequency */
  std::complex<float> w(_eps*_dw,-(_ow + iw*_dw)); // Anti-causal

  /* Apply taper to data */
  apply_taper(dat, slc);

  /* Image at z=0 is just wavefield (data) (just loop and add) */
  for(int iy = 0; iy < _ny; ++iy) {
    for(int ix = 0; ix < _nx; ++ix) {
      img[iy*_nx + ix] += std::real(slc[iy*_nx + ix]);
    }
  }

  /* Loop over depth */
  for(int iz = 0; iz < _nz-1; ++iz) {
    /* Downward continue the data */
    ssr3ssf(w, iz, _slo+(iz)*_nx*_ny, _slo+(iz+1)*_nx*_ny, slc);
    /* Imaging condition */
    for(int iy = 0; iy < _ny; ++iy) {
      for(int ix = 0; ix < _nx; ++ix) {
        img[(iz+1)*_nx*_ny + iy*_nx + ix] += std::real(slc[iy*_nx + ix]);
      }
    }
  }

  /* Free memory */
  delete[] slc;
}

void ssr3::ssr3ssf(std::complex<float> w, int iz, float *scur, float *snex, std::complex<float> *wxin, std::complex<float> *wxot) {

  /* Temporary arrays */
  std::complex<float> *pk  = new std::complex<float>[_bx*_by]();
  std::complex<float> *wk  = new std::complex<float>[_bx*_by]();
  float *wt = new float[_nx*_ny]();

  std::complex<float> w2 = w*w;

  /* w-x-y part 1 */
  for(int iy = 0; iy < _ny; ++iy) {
    for(int ix = 0; ix < _nx; ++ix) {
      float s = 0.5 * scur[iy*_nx + ix];
      wxot[iy*_nx + ix] = wxin[iy*_nx+ix]*exp(-w*s*_dz);
    }
  }

  /* FFT (w-x-y) -> (w-kx-ky) */
  for(int iy = 0; iy < _ny; ++iy) {
    memcpy(&pk[iy*_bx],&wxot[iy*_nx],sizeof(std::complex<float>)*_nx*_ny);
  }
  //plotplt_cmplx(_bx, pk, 0);
  fft2(false,(kiss_fft_cpx*)pk);
  //plotplt_cmplx(_bx, pk, 0);

  memset(wxot,0,sizeof(std::complex<float>)*(_nx*_ny));

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
    //plotplt_cmplx(_bx, wk, 0);
    fft2(true,(kiss_fft_cpx*)wk);
    //plotplt_cmplx(_bx, wk, 0);

    /* Interpolate (accumulate) */
    for(int iy = 0; iy < _ny; ++iy) {
      for(int ix = 0; ix < _nx; ++ix) {
        float d = fabsf(scur[iy*_nx + ix]*scur[iy*_nx + ix] - _sloref[iz*_nrmax + ir]);
        d = _dsmax2/(d*d + _dsmax2);
        wxot[iy*_nx + ix] += wk[iy*_bx + ix]*d;
        wt  [iy*_nx + ix] += d;
      }
    }
  }

  /* w-x-y part 2 */
  for(int iy = 0; iy < _ny; ++iy) {
    for(int ix = 0; ix < _nx; ++ix) {
      wxot[iy*_nx + ix] /= wt[iy*_nx + ix];
      float s = 0.5 * snex[iy*_nx + ix];
      wxot[iy*_nx + ix] *= exp(-w*s*_dz);
    }
  }

  apply_taper(wxot);

  /* Free memory */
  delete[] pk; delete[] wk; delete[] wt;
}

void ssr3::ssr3ssf(std::complex<float> w, int iz, float *scur, float *snex, std::complex<float> *wx) {

  /* Temporary arrays */
  std::complex<float> *pk  = new std::complex<float>[_bx*_by]();
  std::complex<float> *wk  = new std::complex<float>[_bx*_by]();
  float *wt = new float[_nx*_ny]();

  std::complex<float> w2 = w*w;

  /* w-x-y part 1 */
  for(int iy = 0; iy < _ny; ++iy) {
    for(int ix = 0; ix < _nx; ++ix) {
      float s = 0.5 * scur[iy*_nx + ix];
      wx[iy*_nx + ix] *= exp(-w*s*_dz);
    }
  }

  /* FFT (w-x-y) -> (w-kx-ky) */
  for(int iy = 0; iy < _ny; ++iy) {
    memcpy(&pk[iy*_bx],&wx[iy*_nx],sizeof(std::complex<float>)*_nx*_ny);
  }
  fft2(false,(kiss_fft_cpx*)pk);

  memset(wx,0,sizeof(std::complex<float>)*(_nx*_ny));

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

  /* w-x-y part 2 */
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

int ssr3::nrefs(int nrmax, float ds, int ns, float *slo, float *sloref) {

  /* Temporary slowness array */
  float *slo2 = new float[ns]();

  memcpy(slo2,slo,sizeof(float)*ns);

  float smax = quantile(ns-1,ns,slo2);
  float smin = quantile(   0,ns,slo2);
  nrmax = (nrmax < 1 + (smax-smin)/ds) ? (nrmax) : (1+(smax-smin)/ds);

  int jr = 0; float s2 = 0.0;
  for(int ir = 0; ir < nrmax; ++ir) {
    float qr = (ir + 1.0)/nrmax - 0.5 * 1/nrmax;
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

  /* Copy slcin to slcot */
  memcpy(slcot,slcin,sizeof(std::complex<float>)*_nx*_ny);

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
  float dky = 2.0*M_PI/(by*dy); float oky = (by == 1) ? 0 : -M_PI/dy;

  /* Populate the array */
  for(int iy = 0; iy < by; ++iy) {
    int jy = (iy < by/2.) ? (iy + by/2.) : (iy-by/2.);
    float ky = oky + jy*dky;
    for(int ix = 0; ix < bx; ++ix) {
      int jx = (ix < bx/2.) ? (ix + bx/2.) : (ix-bx/2.);
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
    for(int iy = 0; iy < _by; ++iy) {
      kiss_fft(_fwd1, pp + iy*_bx, pp + iy*_bx);
    }

  }

  delete[] ctrace;
}

kiss_fft_cpx ssr3::cmul(kiss_fft_cpx a, float b) {

  a.r *= b;
  a.i *= b;
  return a;
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

void interp_slow(int nz, int nvy, float ovy, float dvy, int nvx, float ovx, float dvx,
                 int ny, float oy, float dy, int nx, float ox, float dx,
                 float *sloin, float *sloot) {

  for(int iz = 0; iz < nz; ++iz) {
    for(int iy = 0; iy < ny; ++iy) {
      float y = oy + iy*dy;
      int icy = (y - ovy)/dvy + 0.5;
      icy = (icy < 0 ) ? 0 : ( ( icy > nvy-1) ? nvy-1 : icy );
      for(int ix = 0; ix < nx; ++ix) {
        float x = ox + ix*dx;
        int icx = (x - ovx)/dvx + 0.5;
        icx = (icx < 0 ) ? 0 : ( ( icx > nvx-1) ? nvx-1 : icx );
        sloot[iz*ny*nx + iy*nx + ix] = sloin[iz*nvy*nvx + icy*nvx + icx];
      }
    }
  }
}

