#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main(int argc, char **argv) {

  int n = 20;
  float *arr = new float[n]();

  for(int i = 10; i < n; ++i) {
    arr[i] = -1.0;
  }

  std::vector<float> v {arr,arr+n};
  plt::plot(v);
  plt::show();
}
