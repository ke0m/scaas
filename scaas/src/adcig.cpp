#include <stdio.h>
#include "adcig.h"
#include "slantstk.h"
#include "tan2ang.h"
#include <string>
//#include "matplotlibcpp.h"

//namespace plt = matplotlibcpp;

void convert2ang(int nro, int nx, int nh, float oh, float dh,
    int nta, float ota, float dta, int na, float oa, float da, int nz, float oz, float dz,
    int ext, float *off, float *ang) {

   //std::map<std::string, std::string> keywords;
   //keywords["aspect"] = "auto"; keywords["cmap"] = "gray";
  /* Build the slant stack operator */
  slantstk ssk = slantstk(true, nh, oh, dh, nta, ota, dta, nz, oz, dz, 0.0, 1.0);

  /* Array for holding tangent gather*/
  float *tan = new float[nta*nz]();

  //TODO: parallelize this loop
  /* Loop over residually migrated images */
  for(int iro = 0; iro < nro; ++iro) {
    /* Loop over image points (CDPs) */
    for(int ix = 0; ix < nx; ++ix) {
      fprintf(stderr,"ix=%d\n",ix);
      /* Grab a single offset gather */
      float *gat = off + iro*nx*nh*nz + ix*nh*nz;
      //plt::imshow((const float*)gat, nh, nz, 1,keywords); plt::show();
      /* Create the tangent gather */
      ssk.adjoint(false, nta*nz, nh*nz, tan, gat);
      //plt::imshow((const float*)tan, nta, nz, 1,keywords); plt::show();
      /* Convert to angle */
      tan2ang(nz, nta, ota, dta, na, oa, da, ext, tan, ang + iro*nx*na*nz + ix*na*nz);
    }
  }

  /* Free memory */
  delete[] tan;
}
