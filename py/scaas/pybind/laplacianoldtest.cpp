extern "C" {
#include "laplacianFWDISPC.h"
}
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main(int argc, char **argv) {

  int nx = 571; int nz = 261;
  float *fld = new float[nx*nz]();
  float *lap = new float[nx*nz]();

  fld[(20)*nx + 50] = 1;

  laplacianFWDISPC(nx,nz,1,1,fld,lap);
  plt::imshow((const float*)fld,nx,nz,1);
  plt::imshow((const float*)lap,nx,nz,1);
  plt::show();

  delete[] fld; delete[] lap;

}
