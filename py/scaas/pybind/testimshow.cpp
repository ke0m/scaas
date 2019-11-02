#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main(int argc, char **argv) {

  int nx = 20; int nz = 20;
  float *arr = new float[nx*nz]();

  for(int ix = 0; ix < nx; ++ix) {
    for(int iz = 10; iz < nz; ++iz) {
      arr[ix*nz + iz] = 1.0;
    }
  }
  
  const int colors = 1;
  plt::imshow((const float*)arr,nz,nx,colors);
  plt::show();

}
