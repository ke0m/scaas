#include <fftw3.h>
#include <cstring>
#include <omp.h>
#include "adcigkzkx.h"
#include "progressbar.h"

void convert2angkzkykx(int ngat,
                       int nz, float oz, float dz,
                       int nhy, float ohy, float dhy,
                       int nhx, float ohx, float dhx,
                       float oa, float da,
                       std::complex<float> *off, std::complex<float> *ang,
                       float eps, int nthrds, bool verb) {

  /*  Allocate memory */
  std::complex<float> *offkzkx = new std::complex<float>[ngat*nz*nhy*nhx]();
  std::complex<float> **angkzs  = new std::complex<float>*[nthrds]();
  std::complex<float> **angzs   = new std::complex<float>*[nthrds]();

  /* FFTW plans */
  int rankf = 3; int nf[] = {nz,nhy,nhx}; int howmanyf = ngat;
  int idistf = nz*nhy*nhx; int odistf = idistf;
  int istridef = 1; int ostridef = istridef;
  int *inembedf = nf, *onembedf = nf;
  fftwf_plan fplan = fftwf_plan_many_dft(rankf,nf,howmanyf,
                                         reinterpret_cast<fftwf_complex*>(off),
                                         inembedf, istridef, idistf,
                                         reinterpret_cast<fftwf_complex*>(offkzkx),
                                         onembedf,ostridef,odistf,
                                         FFTW_FORWARD,FFTW_MEASURE);

  /* Arguments for inverse FFTW */
  int ranki = 1; int ni[] = {nz}; int howmanyi = nhy*nhx;
  int idisti = 1; int odisti = 1;
  int istridei = nhy*nhx; int ostridei = istridei;
  int *inembedi = ni, *onembedi = ni;

  /* FFTW Inverse plans */
  fftwf_plan *iplans = new fftwf_plan[nthrds]();
  /* Inverse FFT scaling */
  float iscale = 1/sqrt(nz);

  /* Khx axis */
  float okhx = -static_cast<int>(nhx/2 + 0.5) * 2*M_PI/(nhx*dhx);
  float dkhx = 2*M_PI/(nhx*dhx);
  /* Complex tridiagonal solver for each thread */
  ctrist **solves = new ctrist*[nthrds]();

  /* Allocate memory for each thread */
  for(int ithrd = 0; ithrd < nthrds; ++ithrd) {
    /* Temporary arrays */
    angkzs[ithrd] = new std::complex<float>[nz*nhy*nhx]();
    angzs [ithrd] = new std::complex<float>[nz*nhy*nhx]();
    /* Inverse FFTW Plans */
    iplans[ithrd] =  fftwf_plan_many_dft(ranki,ni,howmanyi,
                                         reinterpret_cast<fftwf_complex*>(angkzs[ithrd]),
                                         inembedi,istridei,idisti,
                                         reinterpret_cast<fftwf_complex*>(angzs [ithrd]),
                                         onembedi,ostridei,odisti,
                                         FFTW_BACKWARD,FFTW_MEASURE);
    /* Complex tridiagonal solvers */
    solves[ithrd] = new ctrist(nhx,okhx,dkhx,eps);
  }

  /* Compute 3D FFT over all gathers */
  fftwf_execute(fplan);

  /* Vertical wavenumber (kz) */
  float *kzi = new float[nz]();
  int nzh = (int)nz/2 + 0.5;
  for(int iz = 0; iz < nzh; ++iz) {
    kzi[iz    ] = iz;
    kzi[iz+nzh] = (-nzh+iz);
  }

  /* Compute mapping from khx to angle */
  float *stretch = new float[nz*nhx]();
  for(int iz = 0; iz < nz; ++iz) {
    float kz = 2*M_PI * kzi[iz]/(nz*dz);
    for(int ia = 0; ia < nhx; ++ia) {
      float a = oa + ia*da;
      float ka = M_PI * a/180.0;
      float khx = -kz * tanf(ka);
      stretch[iz*nhx + ia] = khx;
    }
  }

  /* Verbosity */
  int *gidx = new int[nthrds]();
  int csize = (int)ngat/nthrds;
  bool firstiter = true;

  /* Loop over gathers */
  omp_set_num_threads(nthrds);
#pragma omp parallel for default(shared)
  for(int igat = 0; igat < ngat; ++igat) {
    int gthd = omp_get_thread_num();
    /* Verbosity */
    if(firstiter && verb) gidx[gthd] = igat;
    if(verb) printprogress_omp("ngat", igat-gidx[gthd], csize, gthd);
    /* Apply forward shift */
    forwardshift(nz,oz,dz,nhy,ohy,dhy,nhx,ohx,dhx,offkzkx + igat*nz*nhy*nhx);
    /* Offset to angle conversion */
    memset(angkzs[gthd],0,sizeof(std::complex<float>)*nz*nhy*nhx);
    convertone2angkhx(nz,nhx,okhx,dkhx,nhy,stretch,eps,solves[gthd],
                       offkzkx + igat*nz*nhy*nhx, angkzs[gthd]);
    /* Apply Inverse shift */
    inverseshift(nz, oz, dz, nhy, nhx, angkzs[gthd]);
    /* Inverse FFT along z for this gather */
    fftwf_execute(iplans[gthd]);
    /* Apply inverse scale */
    for(int k = 0; k < nz*nhy*nhx; ++k) angzs[gthd][k] *= iscale;
    /* Copy to output angle */
    memcpy(&ang[igat*nhx*nhy*nz],angzs[gthd],sizeof(std::complex<float>)*nhy*nhx*nz);
    /* Verbosity */
    firstiter = false;
  }
  if(verb) printf("\n");

  /* Destroy plans and free memory */
  fftwf_destroy_plan(fplan);
  delete[] offkzkx; delete[] kzi; delete[] stretch;
  delete[] gidx;
  for(int ithrd = 0; ithrd < nthrds; ++ithrd) {
    fftwf_destroy_plan(iplans[ithrd]);
    delete[] angkzs[ithrd]; delete[] angzs[ithrd];
    delete solves[ithrd];
  }
  delete[] iplans; delete[] angkzs; delete[] angzs;
  delete[] solves;
}

void convertone2angkhx(int nz, int nkhx, float okhx, float dkhx, int nkhy,
                       float* stretch, float eps, ctrist *solv,
                       std::complex<float> *off, std::complex<float> *ang) {

  /* Temporary array */
  std::complex<float> *data = new std::complex<float>[nkhx]();
  int jhx = (int)nkhx/2 + 0.5;

  /* Loop over depth */
  for(int iz = 0; iz < nz; ++iz) {
    /* Build the linear system with ctrist define */
    solv->define(stretch + iz*nkhx);
    /* Interp for all ys */
    for(int ihy = 0; ihy < nkhy; ++ihy) {
      /* Get the data for this depth */
      memcpy(&data[0  ],&off[iz*nkhy*nkhx + ihy*nkhx + jhx],sizeof(std::complex<float>)*jhx);
      memcpy(&data[jhx],&off[iz*nkhy*nkhx + ihy*nkhx +   0],sizeof(std::complex<float>)*jhx);
      /* Perform the inverse interpolation */
      solv->apply(ang + iz*nkhy*nkhx + ihy*nkhx, data);
    }
  }

  /* Free memory */
  delete[] data;
}

void forwardshift(int nz, float oz, float dz,
                  int nhy, float ohy, float dhy,
                  int nhx, float ohx, float dhx,
                  std::complex<float>* data) {

  float scale = 1/sqrtf(nz*nhy*nhx);

  for(int iz = 0; iz < nz; ++iz) {
    for(int ihy = 0; ihy < nhy; ++ihy) {
      for(int ihx = 0; ihx < nhx; ++ihx) {
        float argF = 2*M_PI * ((float)iz/(float)nz*oz/dz + (float)ihx/(float)nhx*ohx/dhx + (float)ihy*(float)nhy*ohy/dhy);
        data[iz*nhx*nhy + ihy*nhx + ihx] *= std::complex<float>(cosf(argF),-(+1)*sinf(argF))*scale;
      }
    }
  }
}

void inverseshift(int nz, float oz, float dz,
                  int nhy, int nhx, std::complex<float> *data) {

  for(int iz = 0; iz < nz; ++iz) {
    for(int ihy = 0; ihy < nhy; ++ihy) {
      for(int ihx = 0; ihx < nhx; ++ihx) {
        float argI = 2*M_PI * ((float)iz/(float)nz*oz/dz);
        data[iz*nhx*nhy + ihy*nhx + ihx] *= std::complex<float>(cosf(argI),-(-1)*sinf(argI));
      }
    }
  }
}
