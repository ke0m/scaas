#include <omp.h>

void myompfunc(int n, float scale, float *in, float *ot, int nthreads) {

  omp_set_num_threads(nthreads);
#pragma omp parallel for default(shared)
  for(int i = 0; i < n; ++i) {
    ot[i] = in[i]*scale;
  }
}
