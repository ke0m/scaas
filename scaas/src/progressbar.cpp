#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <string>

#define PBSTR "============================================================="
#define PBWIDTH 60

void printprogress(std::string prefix, int icur, int tot) {
  double percentage = (double)icur/tot;
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad-1;
  printf ("\r%s [%.*s>%*s] %d/%d", prefix.c_str(), lpad, PBSTR, rpad, "", icur, tot);
  fflush (stdout);
  if(icur == tot-1) {
    printf ("\r%s [%.*s%*s] %d/%d", prefix.c_str(), PBWIDTH, PBSTR, 0, "", tot, tot);
    printf("\n");
  }
}

void printprogress_omp(std::string prefix, int icur, int tot, int thread) {
  double percentage = (double)icur/tot;
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad-1;
  std::string tid;
  if(thread < 10) {
    tid = std::string( 1, '0').append(std::to_string(thread));
  } else{
    tid = std::to_string(thread);
  }
  printf ("\r(thd: %s) %s [%.*s>%*s] %d/%d", tid.c_str(), prefix.c_str(), lpad, PBSTR, rpad, "", icur, tot);
  fflush (stdout);
  if(icur == tot-1) {
    printf ("\r(thd: %s) %s [%.*s%*s] %d/%d", tid.c_str(), prefix.c_str(), PBWIDTH, PBSTR, 0, "", tot, tot);
    printf(" ");
  }
}

int main(int argc, char **argv) {

  int k1 = 0, k2 = 0;
  int nthd = 4, ctr = 0;
  int *sidx = new int[nthd]();
  int tot = 21;
  omp_set_num_threads(nthd);
  // compute the maximum chunk size
  int csize = (int)tot/nthd;
  if(tot%nthd != 0) csize += 1;
  bool firstiter = true;
#pragma omp parallel for default(shared)
  for(int i = 0; i < tot; ++i) {
    if(firstiter) {
      sidx[omp_get_thread_num()] = i;
    }
    usleep(100000);
    printprogress_omp("nshots:", i - sidx[omp_get_thread_num()], csize, omp_get_thread_num());
    firstiter = false;
    //printProgress("nshots", i, 100);
  }
  printf("\n");
  delete[] sidx;
}
