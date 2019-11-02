#include <stdio.h>
#include <cstring>

int main(int argc, char **argv) {

  int n = 100;
  float *test = new float[n]();
  float *agan = new float[n]();

  for(int k = 0; k < 100; ++k) {
    test[k] = static_cast<float>(k);
  }

  memcpy(agan,test,sizeof(float)*n);
  fprintf(stderr,"%f\n",agan[20]);

  delete[] test; delete[] agan;

}
