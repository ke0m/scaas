#include <math.h>
#include "halfint.h"

halfint::halfint(bool inv, int n1, float rho) {
  _n  = 2*kiss_fft_next_fast_size((n1+1)/2);
  _nw = _n/2+1;

  _cx = new kiss_fft_cpx[_nw]();
  _cf = new kiss_fft_cpx[_nw]();

  _forw = kiss_fftr_alloc(_n,0,NULL,NULL);
  _invs = kiss_fftr_alloc(_n,1,NULL,NULL);

  if(NULL == _forw || NULL == _invs)
    fprintf(stderr,"KISS FFT allocation error");

  kiss_fft_cpx cw, cz, cz2;

  /* Construct the filter in the frequency domain */
  for(int i = 0; i < _nw; ++i) {
    float om = -2*M_PI*i/_n;
    cw.r = cosf(om);
    cw.i = sinf(om);

    cz.r = 1.0 - rho*cw.r;
    cz.i = -rho*cw.i;
    if(inv) {
      cz = csqrtf(cz);
    } else {
      cz2.r = 0.5*(1 + rho*cw.r);
      cz2.i = 0.5*rho*cw.i;
      cz = csqrtf(cdiv(cz2,cz));
    }
    _cf[i].r = cz.r/_n;
    _cf[i].i = cz.i/_n;
  }
}

kiss_fft_cpx halfint::csqrtf(kiss_fft_cpx c) {

  float d, r, s;
  kiss_fft_cpx v;

  if (c.i == 0) {
    if (c.r < 0) {
      v.r = 0.;
      v.i = copysignf (sqrtf (-c.r), c.i);
    } else {
      v.r =  fabsf (sqrtf (c.r));
      v.i =  copysignf (0, c.i);
    }
  } else if (c.r == 0) {
    r = sqrtf (0.5 * fabsf (c.i));
    v.r = r;
    v.i = copysignf (r, c.i);
  } else {
    d = hypotf (c.r, c.i);
    /* Use the identity   2  Re res  Im res = Im x
       to avoid cancellation error in  d +/- Re x.  */
    if (c.r > 0) {
      r = sqrtf (0.5f * d + 0.5f * c.r);
      s = (0.5f * c.i) / r;
    } else {
      s = sqrtf (0.5f * d - 0.5f * c.r);
      r = fabsf ((0.5f * c.i) / s);
    }
    v.r = r;
    v.i = copysignf (s, c.i);
  }
  return v;
}

kiss_fft_cpx halfint::cdiv(kiss_fft_cpx a, kiss_fft_cpx b) {

  kiss_fft_cpx c;
  float r,den;
  if (fabsf(b.r)>=fabsf(b.i)) {
    r = b.i/b.r;
    den = b.r+r*b.i;
    c.r = (a.r+r*a.i)/den;
    c.i = (a.i-r*a.r)/den;
  } else {
    r = b.r/b.i;
    den = b.i+r*b.r;
    c.r = (a.r*r+a.i)/den;
    c.i = (a.i*r-a.r)/den;
  }
  return c;
}


