#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "aainterp.h"

aainterp::aainterp(int n1, float o1, float d1, int n2) {
  _n1 = n1; _n2 = n2; _nk = 3;
  _o1 = o1; _d1 = d1;
}

void aainterp::define(const float *coord, const float *delt, const float *amp,
    float *x, bool *m, float *w, float *a) {

  int ix[3] = {0};
  float rx[3] = {0.0};

  for (int id = 0; id < _n2; id++) {
    m[id] = false;

    rx[0] = coord[id] + delt[id] + _d1;
    rx[1] = coord[id];
    rx[2] = coord[id] - delt[id] - _d1;
    for (int j=0; j < _nk; j++) {
      rx[j] = (rx[j] - _o1)/_d1;
      ix[j] = rx[j];
      if ((ix[j] < 0) || (ix[j] > _n1 - 2)) {
        m[id] = true;
        break;
      }
      w[id*_nk + j] = rx[j] - ix[j];
    }
    if (m[id]) continue;

    x[id*_nk + 0] = ix[0];
    x[id*_nk + 1] = ix[1] +   _n1;
    x[id*_nk + 2] = ix[2] + 2*_n1;
    a[id] = _d1*_d1/(delt[id]*delt[id] + _d1*_d1);

    if (nullptr != amp) a[id] *= amp[id];
  }

}

void aainterp::forward(bool add, int n1, int n2, const float *x, const bool *m, const float *w, const float *a,
    float *ord, float *mod) {

  if (n1 != _n1 || n2 != _n2) {
    fprintf(stderr,"Sizes passed to forward do not match constructor");
    exit(EXIT_FAILURE);
  }

  if(!add) memset(mod, 0, sizeof(float)*_n1);

  /* Temporary arrays */
  float *tmp1 = new float[_n1*_nk]();
  float *tmp2 = new float[_n1]();

  for(int id = 0; id < _n2; ++id) {
    if(m[id]) continue;

    float aa = a[id];
    for(int j = 0; j < _nk; ++j) {
      int i1 = x[id*_nk + j];
      int i2 = i1 + 1;

      float w2 = w[id*_nk + j];
      float w1 = 1.0 - w2;

      w2 *= aa;
      w1 *= aa;

      tmp1[i1] += w1 * ord[id];
      tmp1[i2] += w2 * ord[id];
    }
  }

  for(int it = 0; it < _n1; ++it) {
    tmp2[it] = 2*tmp1[it+_n1] - tmp1[it] - tmp1[it+2*_n1];
  }

  doubint(true,_n1,tmp2);

  for(int it = 0; it < _n1; ++it) {
    mod[it] += tmp2[it];
  }

  /* Free memory */
  delete[] tmp1; delete[] tmp2;
}

void aainterp::adjoint(bool add, int n1, int n2, const float *x, const bool *m, const float *w, const float *a,
    float *ord, float *mod) {

  if (n1 != _n1 || n2 != _n2) {
    fprintf(stderr,"Sizes passed to forward do not match constructor");
    exit(EXIT_FAILURE);
  }

  if(!add) memset(ord, 0, sizeof(float)*_n2);

  /* Temporary arrays */
  float *tmp1 = new float[_n1*_nk]();
  float *tmp2 = new float[_n1]();

  memcpy(tmp2,mod,sizeof(float)*_n1);

  doubint (true, _n1, tmp2);

  for (int it = 0; it < _n1; it++) {
    tmp1[it+_n1]   =  tmp2[it]*(_nk-1);
    tmp1[it]       = -tmp2[it];
    tmp1[it+2*_n1] = -tmp2[it];
  }

  for(int id = 0; id < _n2; ++id) {
    if(m[id]) continue;

    float aa = a[id];
    for(int j = 0; j < _nk; ++j) {
      int i1 = x[id*_nk + j];
      int i2 = i1 + 1;

      float w2 = w[id*_nk + j];
      float w1 = 1. - w2;

      w2 *= aa;
      w1 *= aa;

      ord[id] += w2 * tmp1[i2] + w1 * tmp1[i1];
    }
  }

  /* Free memory */
  delete[] tmp1; delete[] tmp2;
}

void aainterp::doubint(bool dble, int n, float *trace) {

  float t = 0.;
  for (int it = 0; it < n; it++) {
    t += trace[it];
    trace[it] = t;
  }

  if (dble) {
    t = 0.;
    for (int it = n-1; it >=0; it--) {
      t += trace[it];
      trace[it] = t;
    }
  }

}
