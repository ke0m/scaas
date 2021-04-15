#include <math.h>
#include <cstring>
#include "ctrist.h"

ctrist::ctrist(int n, float o, float d, float eps) {
  _nm = n; _om = o; _dm = d; _eps = eps;

  /* Allocate memory */
  _idx  = new int[_nm]();
  _flg  = new bool[_nm]();
  _wgt  = new std::complex<float>[_nm]();
  _diag = new std::complex<float>[_nm]();
  _offd = new std::complex<float>[_nm]();
  /* Temporary arrays */
  _dd = new std::complex<float>[_nm]();
  _oo = new std::complex<float>[_nm]();
}

void ctrist::define(float *coord) {
  /* Builds the tridiagonal system */

  /* Initialize diag and offdiag to zero */
  memset(_diag,0,sizeof(std::complex<float>)*_nm);
  memset(_offd,0,sizeof(std::complex<float>)*_nm);

  for(int im = 0; im < _nm; ++im) {
    float rm = (coord[im]-_om)/_dm;
    int il = (int)rm;
    float wu = rm - il;

    if(il < 0 || il >= _nm-1 || wu < 0) {
      _flg[im] = true;
      continue;
    }

    _flg[im] = false;
    _idx[im] = il;
    _wgt[im] = std::complex<float>(wu,0);

    float wl = 1.0 - wu;

    _diag[im] += std::complex<float>(wl*wl,0);
    _diag[im] += std::complex<float>(wu*wu,0);
  }

  /* Add regularization parameter */
  for(int im = 0; im < _nm; ++im) {
    _diag[im] += 2*_eps;
    _offd[im] -=   _eps;
  }

}

void ctrist::apply(std::complex<float> *model, std::complex<float> *data) {

  for(int im = 0; im < _nm; ++im) {
    if(_flg[im]) continue;

    int il = _idx[im];
    std::complex<float> wu = _wgt[im];

    int iu = il+1;
    std::complex<float> wl = std::complex<float>(1.0,0) - wu;

    model[im] += data[il] * wl;
    model[im] += data[iu] * wu;
  }

  solve(_diag,_offd,model);

}

void ctrist::solve(std::complex<float> *diag, std::complex<float> *offd,
                                              std::complex<float> *model) {
  _dd[0] = diag[0];
  for(int k = 1; k < _nm; ++k) {
    std::complex<float> t = offd[k-1];
    _oo[k-1] = t/diag[k-1];
    _dd[k  ] = diag[k] - t*_oo[k-1];
  }

  for(int k = 1; k < _nm; ++k) {
    model[k] -= _oo[k-1]*model[k-1];
  }

  model[_nm-1] /= _dd[_nm-1];

  for(int k = _nm-2; k >= 0; --k) {
    model[k] = model[k]/_dd[k] - _oo[k]*model[k+1];
  }

}
