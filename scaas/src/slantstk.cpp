#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include "halfint.h"
#include "aainterp.h"
#include "slantstk.h"

slantstk::slantstk(bool rho,
    int nx, float ox, float dx,
    int ns, float os, float ds,
    int nt, float ot, float dt, float s11, float anti1) {
  _rho = rho;
  /* Get axes */
  _nx = nx; _ox = ox; _dx = dx;
  _ns = ns; _os = os; _ds = ds;
  _nt = nt; _ot = ot; _dt = dt;

  _s11 = s11; _anti1 = anti1;
}

void slantstk::forward(bool add, int nm, int nd, float *mod, float *dat) {

  if(nm != _nt*_ns || nd != _nt*_nx) {
    fprintf(stderr,"slantstk: Dimensions don't match passed to constructor.\n");
    exit(EXIT_FAILURE);
  }

  if(!add) memset(dat, 0, sizeof(float)*nd);

  /* Allocate temporary arrays */
  float *amp = new float[_nt]();
  float *str = new float[_nt]();
  float *tx  = new float[_nt]();
  float *tmp = new float[_nt]();
  /* Arrays for aaop */
  bool *miss = new bool[_nt]();
  float *xin = new float[3*_nt]();
  float *win = new float[3*_nt]();
  float *ain = new float[_nt]();

  /* Create operators */
  aainterp aaop = aainterp(_nt, _ot, _dt, _nt);
  halfint  hiop = halfint(true,2*_nt,1.-1./_nt);

  for(int is = 0; is < _ns; ++is) { /* Slowness */
    float s =  _os + is*_ds;
    for(int ix = 0; ix < _nx; ++ix) { /* Offset */
      float x = _ox + ix*_dx;
      float sxx = s*x;
      /* Build str,tx and amp arrays */
      for(int it = 0; it < _nt; ++it) { /* Time */
        float z = _ot + it*_dt;
        float t = z + sxx;

        str[it] = t;
        tx[it]  = _anti1*fabsf(s-_s11)*_dx;
        amp[it] = 1.;
      }
      /* Populates the xin, miss, win and ain arrays */
      aaop.define(str, tx, amp, xin, miss, win, ain);
      if(_rho) {
        aaop.forward(false, _nt, _nt, xin, miss, win, ain, mod + is*_nt, tmp);
        hiop.adjoint(true , _nt, dat + ix*_nt, tmp);
      } else {
        aaop.forward(true, _nt, _nt, xin, miss, win, ain, mod + is*_nt , dat + ix*_nt);
      }
    }
  }

  /* Free memory */
  delete[] miss; delete[] xin; delete[] win; delete[] ain;
  delete[] amp; delete[] str; delete[] tx; delete[] tmp;
}

void slantstk::adjoint(bool add, int nm, int nd, float *mod, float *dat) {

  if(nm != _nt*_ns || nd != _nt*_nx) {
    fprintf(stderr,"slantstk: Dimensions don't match passed to constructor.\n");
    exit(EXIT_FAILURE);
  }

  if(!add) memset(mod, 0, sizeof(float)*nm);

  /* Allocate temporary arrays */
  float *amp = new float[_nt]();
  float *str = new float[_nt]();
  float *tx  = new float[_nt]();
  float *tmp = new float[_nt]();
  /* Arrays for aaop */
  bool *miss = new bool[_nt]();
  float *xin = new float[3*_nt]();
  float *win = new float[3*_nt]();
  float *ain = new float[_nt]();

  /* Create operators */
  aainterp aaop = aainterp(_nt, _ot, _dt, _nt);
  halfint  hiop = halfint(true,2*_nt,1.-1./_nt);

  for(int is = 0; is < _ns; ++is) { /* Slowness */
    float s =  _os + is*_ds;
    for(int ix = 0; ix < _nx; ++ix) { /* Offset */
      float x = _ox + ix*_dx;
      float sxx = s*x;
      /* Build str,tx and amp arrays */
      for(int it = 0; it < _nt; ++it) { /* Time */
        float z = _ot + it*_dt;
        float t = z + sxx;

        str[it] = t;
        tx[it]  = _anti1*fabsf(s-_s11)*_dx;
        amp[it] = 1.;
      }
      /* Populates the xin, miss, win and ain arrays */
      aaop.define(str, tx, amp, xin, miss, win, ain);
      if(_rho) {
        hiop.forward(false, _nt, dat + ix*_nt, tmp);
        aaop.adjoint(true , _nt, _nt, xin, miss, win, ain, mod + is*_nt, tmp);
      } else {
        aaop.adjoint(true, _nt, _nt, xin, miss, win, ain, mod + is*_nt , dat + ix*_nt);
      }
    }
  }

  /* Free memory */
  delete[] miss; delete[] xin; delete[] win; delete[] ain;
  delete[] amp; delete[] str; delete[] tx; delete[] tmp;

}
