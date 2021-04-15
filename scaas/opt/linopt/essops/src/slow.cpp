#include <math.h>
#include <stdio.h>
#include "slow.h"

void slowforward(int nq, float oq, float dq, int nz, float oz, float dz,
                 int nx, float ox, float dx, int nt, float ot, float dt,
                 float *mod, float *dat) {

  for(int iq = 0; iq < nq; ++iq) {
    float q = oq + iq*dq;
    for(int ix = 0; ix < nx; ++ix) {
      float x = ox + ix*dx;
      float sx = fabsf(q * x);
      for(int iz = 0; iz < nz; ++iz) {
        float z = oz + iz*dz;
        float t = sqrtf( z * z + sx * sx);
        /* Compute linear interpolation weights */
        float f = (t - oz)/dz;
        int it  = static_cast<int>(f + 0.5);
        float fx = f   - it;
        float gx = 1.0 - fx;
        if(it >= 0 && it < nt-1) {
          dat[ix*nt + it+0] += gx*mod[iq*nz + iz];
          dat[ix*nt + it+1] += fx*mod[iq*nz + iz];
        }
      }
    }
  }
}

void slowadjoint(int nq, float oq, float dq, int nz, float oz, float dz,
                 int nx, float ox, float dx, int nt, float ot, float dt,
                 float *mod, float *dat) {

  for(int iq = 0; iq < nq; ++iq) {
    float q = oq + iq*dq;
    for(int ix = 0; ix < nx; ++ix) {
      float x  = ox + ix*dx;
      float sx = fabsf(q * x);
      for(int iz = 0; iz < nz; ++iz) {
        float z = oz + iz*dz;
        float t = sqrtf( z * z + sx * sx);
        /* Compute linear interpolation weights */
        float f = (t - oz)/dz;
        int it  = static_cast<int>(f + 0.5);
        float fx = f   - it;
        float gx = 1.0 - fx;
        if(it >= 0 && it < nt-1) {
          mod[iq*nz + iz] += gx*dat[ix*nt + it+0] + fx*dat[ix*nt + it+1];
        }
      }
    }
  }
}
