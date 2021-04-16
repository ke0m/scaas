#include <omp.h>
#include <cstring>
#include "adcig.h"
#include "slantstk.h"
#include "tan2ang.h"
#include "progressbar/progressbar.h"

void convert2ang(int nx, int nh, float oh, float dh,
    int nta, float ota, float dta, int na, float oa, float da, int nz, float oz, float dz,
    int ext, float *off, float *ang, int nthrd, bool verb) {

  /* Build the slant stack operator */
  slantstk ssk = slantstk(true, nh, oh, dh, nta, ota, dta, nz, oz, dz, 0.0, 1.0);

  /* Array for holding gathers*/
  float *gat = new float[nh *nz*nthrd]();
  float *tan = new float[nta*nz*nthrd]();

  /* Set up printing if verbosity is desired */
  int *cidx = new int[nthrd]();
  int csize = (int)nx/nthrd;
  if(nx%nthrd != 0) csize += 1;
  bool firstiter = true;

  omp_set_num_threads(nthrd);
#pragma omp parallel for default(shared)
  /* Loop over image points (CDPs) */
  for(int ix = 0; ix < nx; ++ix) {
    /* Get thread index */
    int tidx = omp_get_thread_num();
    /* Set up printing */
    if(firstiter && verb) cidx[tidx] = ix;
    if(verb) printprogress_omp("cdp:", ix - cidx[tidx], csize, tidx);
    /* Grab a single offset gather */
    memcpy(&gat[tidx*nh*nz],&off[ix*nh*nz],sizeof(float)*nh*nz);
    /* Create the tangent gather */
    ssk.adjoint(false, nta*nz, nh*nz, tan + tidx*nta*nz, gat + tidx*nh*nz);
    /* Convert to angle */
    tan2ang(nz, nta, ota, dta, na, oa, da, ext, tan + tidx*nta*nz, ang + ix*na*nz);
    /* Parallel printing */
    if(verb) firstiter = false;
  }
  /* Parallel printing */
  if(verb) printf("\n");

  /* Free memory */
  delete[] tan; delete[] gat; delete[] cidx;
}
