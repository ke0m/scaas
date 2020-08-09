#include <fftw3.h>
#include <cstring>
#include <omp.h>
#include "adcigkzkx.h"
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
        tmp[i1*n2 + i2] = real(arr[i1*n2 + i2]);
      } else if(option == 1) {
        tmp[i1*n2 + i2] = imag(arr[i1*n2 + i2]);
      } else {
        tmp[i1*n2 + i2] = abs(arr[i1*n2 + i2]);
      }
    }
  }
  std::map<std::string,std::string> vals;
  //vals["vmax"] = "0.01"; //vals["vmin"] = "0.0";
  vals["cmap"] = "gray"; vals["aspect"] = "auto"; vals["interpolation"] = "sinc";
  plt::imshow((const float *)tmp,n1,n2,1,vals); plt::show();

  delete [] tmp;
}

void plotplt_cmplx(int n1, std::complex<float> *arr, int option) {
  float * tmp = new float[n1];

  for(int i1 = 0; i1 < n1; ++i1) {
    if(option == 0) {
      tmp[i1] = real(arr[i1]);
    } else if(option == 1) {
      tmp[i1] = imag(arr[i1]);
    } else {
      tmp[i1] = abs(arr[i1]);
    }
  }
  std::vector<float> v {tmp,tmp+n1};
  plt::plot(v); plt::show();

  delete [] tmp;

}

void convert2angkzkykx(int ngat,
                       int nz, float oz, float dz,
                       int nhy, float ohy, float dhy,
                       int nhx, float ohx, float dhx,
                       float oa, float da,
                       std::complex<float> *off, std::complex<float> *ang,
                       float eps, int nthrd, bool verb) {

  /*  Allocate memory */
  std::complex<float> *offkzkx = new std::complex<float>[ngat*nz*nhy*nhx]();
  std::complex<float> *angkz   = new std::complex<float>[nz*nhy*nhx]();
  std::complex<float> *angz    = new std::complex<float>[nz*nhy*nhx]();

  /* FFTW plans*/
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

  int ranki = 1; int ni[] = {nz}; int howmanyi = nhy*nhx;
  int idisti = 1; int odisti = 1;
  int istridei = nhy*nhx; int ostridei = istridei;
  int *inembedi = ni, *onembedi = ni;

  fftwf_plan iplan = fftwf_plan_many_dft(ranki,ni,howmanyi,
                                         reinterpret_cast<fftwf_complex*>(angkz),
                                         inembedi,istridei,idisti,
                                         reinterpret_cast<fftwf_complex*>(angz),
                                         onembedi,ostridei,odisti,
                                         FFTW_BACKWARD,FFTW_MEASURE);

  /* Compute 3D FFT over all gathers */
  fftwf_execute(fplan);


  /* Vertical wavenumber (kz) */
  float *kzi = new float[nz]();
  int nzh = (int)nz/2 + 0.5;
  for(int iz = 0; iz < nzh; ++iz) {
    kzi[iz    ] = iz;
    kzi[iz+nzh] = (-nzh+iz);
  }

  /* Khx axis */
  float okhx = -static_cast<int>(nhx/2 + 0.5) * 2*M_PI/(nhx*dhx);
  float dkhx = 2*M_PI/(nhx*dhx);

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

//  std::map<std::string,std::string> vals;
//  //vals["vmax"] = "0.01"; //vals["vmin"] = "0.0";
//  vals["cmap"] = "jet"; vals["aspect"] = "auto";

  /* Create the solver to be passed */
  ctrist *solv = new ctrist(nhx,okhx,dkhx,eps);
  float iscale = 1/sqrt(nz);

  /* Loop over gathers */
  //TODO: parallelize with OpenMP (will need to create one solv per thread)
  //TODO: put a progressbar here
  for(int igat = 0; igat < ngat; ++igat) {
    /* TODO: Verbosity */
    /* Apply forward shift */
    forwardshift(nz,oz,dz,nhy,ohy,dhy,nhx,ohx,dhx,offkzkx + igat*nz*nhy*nhx);
    /* Offset to angle conversion */
    memset(angkz,0,sizeof(std::complex<float>)*nz*nhy*nhx);
    convertone2angkhx(nz,nhx,okhx,dkhx,nhy,stretch,eps,solv,
                       offkzkx + igat*nz*nhy*nhx, angkz);
    /* Apply Inverse shift */
    inverseshift(nz, oz, dz, nhy, nhx, angkz);
    /* Inverse FFT along z for this gather */
    fftwf_execute(iplan);
    /* Apply inverse scale */
    for(int k = 0; k < nz*nhy*nhx; ++k) angz[k] *= iscale;
    /* Copy to output angle */
    memcpy(&ang[igat*nhx*nhy*nz],angz,sizeof(std::complex<float>)*nhy*nhx*nz);
  }

  /* Destroy plans and free memory */
  fftwf_destroy_plan(fplan); fftwf_destroy_plan(iplan);
  delete solv;
  delete[] offkzkx; delete[] angkz; delete[] angz;
  delete[] kzi; delete[] stretch;
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
