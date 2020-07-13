#include <math.h>
#include "mute.h"

void muteone(int nt, float ot, float dt, bool abs0, bool inner, bool hyper,
             float tp, float slope0, float slopep, float x,
             float *datin, float *datot) {

  if(abs0) x = fabsf(x);
  /* Time */
  for(int it = 0; it < nt; ++it) {
    float t = ot + it*dt;
    datot[it] = datin[it];
    if(hyper) t *= t;
    float wt = t - x * slope0;
    /* If out of region, set to 0 */
    if((inner && wt > 0.) || (!inner && wt < 0.)) {
      datot[it] = 0.0;
    } else {
      /* Apply sine weighting function (taper) */
      wt = t - tp - x * slopep;
      if((inner && wt >= 0.) || (!inner && wt <= 0.)) {
        wt = sinf(0.5*M_PI*(t-x*slope0)/(tp+x*(slopep-slope0)));
        datot[it] = datin[it]*(wt*wt);
      }
    }
  }
}

void muteall(int n3, int n2, float o2, float d2, int n1, float o1, float d1,
             float tp, float t0, float v0, float slope0, float slopep, float x0,
             bool abs, bool inner, bool hyper, bool half,
             float *datin, float *datot) {

  /* Hyperbolic option */
  if(hyper) {
    slope0 *= slope0;
    slopep *= slopep;
  }
  /* Gathers */
  for(int i3 = 0; i3 < n3; ++i3) {
    /* Traces within the gather */
    for(int i2 = 0; i2 < n2; ++i2) {
      float x = o2 + i2*d2 - x0; // Offset
      if(half)  x *= 2;          // half-offset
      if(hyper) x *= x;          // hyperbolic mute
      /* Mute trace by trace */
      int tridx = i3*n1*n2 + i2*n1;
      muteone(n1,o1-t0,d1,abs,inner,hyper,
              tp,slope0,slopep,x,datin+tridx,datot+tridx);
    }
  }
}
