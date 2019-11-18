/**
 * Tests Dave Hale's sinc interpolation function
 * @author: Joseph Jennings
 * @version: 2019.11.09
 */

#include <math.h>
#include "matplotlibcpp.h"
#include "sincfuncs.h"

namespace plt = matplotlibcpp;

int main(int argc, char **argv) {

  /* Create a signal */
  int nt = 1000; float ot = 0.0; float dt = 0.01;
  float *sig = new float[nt]();

  for(int it = 0; it < nt; ++it) {
    float t = ot + it*dt;
    sig[it] = sin(t);
  }

  /* Interpolate to finer sampling */
  int ntf = 2*nt; float dtf = dt*0.5;
  float *tf = new float[ntf]();
  float *isig = new float[ntf]();
  float *fsig = new float[ntf]();
  float *diff = new float[ntf]();
  for(int it = 0; it < ntf; ++it) {
    tf[it] = ot + dtf*it;
    fsig[it] = sin(tf[it]);
  }

  ints8r(nt,dt,ot,sig,0.0,0.0,ntf,tf,isig);

  std::vector<float> v1{sig,sig+nt};
  std::vector<float> v2{fsig,fsig+ntf};
  std::vector<float> v3{isig,isig+ntf};
  plt::plot(v1); plt::plot(v2); plt::plot(v3);
  plt::show();

  /* Free memory */
  delete[] sig; delete[] tf; delete[] isig;
  delete[] fsig; delete[] diff;
}
