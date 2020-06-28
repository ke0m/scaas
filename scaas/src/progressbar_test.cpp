#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <omp.h>
#include <string>
#include "progressbar.h"

//#define PBSTR "============================================================="
//#define PBWIDTH 60
//
//void printprogress(std::string prefix, int icur, int tot) {
//  double percentage = (double)icur/tot;
//  int lpad = (int) (percentage * PBWIDTH);
//  int rpad = PBWIDTH - lpad-1;
//  printf ("\r%s [%.*s>%*s] %d/%d", prefix.c_str(), lpad, PBSTR, rpad, "", icur, tot);
//  fflush (stdout);
//  if(icur == tot-1) {
//    printf ("\r%s [%.*s%*s] %d/%d", prefix.c_str(), PBWIDTH, PBSTR, 0, "", tot, tot);
//    printf("\n");
//  }
//}
//
//void printprogress_omp(std::string prefix, int icur, int tot, int thread) {
//  double percentage = (double)icur/tot;
//  int lpad = (int) (percentage * PBWIDTH);
//  int rpad = PBWIDTH - lpad-1;
//  std::string tid;
//  if(thread < 10) {
//    tid = std::string( 1, '0').append(std::to_string(thread));
//  } else{
//    tid = std::to_string(thread);
//  }
//  printf ("\r(thd: %s) %s [%.*s>%*s] %d/%d", tid.c_str(), prefix.c_str(), lpad, PBSTR, rpad, "", icur, tot);
//  fflush (stdout);
//  if(icur == tot-1) {
//    printf ("\r(thd: %s) %s [%.*s%*s] %d/%d", tid.c_str(), prefix.c_str(), PBWIDTH, PBSTR, 0, "", tot, tot);
//    printf(" ");
//  }
//}

//void printprogress_strm(std::string prefix, int icur, int tot) {
//  int barWidth = 70;
//
//  double progress = (double)icur/tot;
//  std::cout << prefix << " " << "[";
//  int pos = barWidth * progress;
//  for (int i = 0; i < barWidth; ++i) {
//    if (i < pos) std::cout << "=";
//    else if (i == pos) std::cout << ">";
//    else std::cout << " ";
//  }
//  std::cout << "] " << std::to_string(icur) << "/" << std::to_string(tot) << "\r";
//  std::cout.flush();
//  if(icur == tot-1) {
//    std::cout << prefix << " " << "[";
//    for (int i = 0; i < barWidth; ++i) std::cout << "=";
//    std::cout << "] " << std::to_string(tot) << "/" << std::to_string(tot) << "\r";
//    std::cout << std::endl;
//  }
//
//}

void printprogress_strm(std::string prefix, int icur, int tot) {
  char buf[PBWIDTH + 20];
  double percentage = (double)icur/tot;
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad-1;
  sprintf (buf,"\r%s [%.*s>%*s] %d/%d", prefix.c_str(), lpad, PBSTR, rpad, "", icur, tot);
  std::string strouti = buf;
  std::cout << strouti;
  std::cout.flush();
  if(icur == tot-1) {
    sprintf (buf,"\r%s [%.*s%*s] %d/%d", prefix.c_str(), PBWIDTH, PBSTR, 0, "", tot, tot);
    std::string stroutf = buf;
    std::cout << stroutf;
    std::cout << std::endl;
  }
}

void printprogress_strm_omp(std::string prefix, int icur, int tot, int thread) {
  char buf[PBWIDTH + 20];
  double percentage = (double)icur/tot;
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad-1;
  std::string tid;
  if(thread < 10) {
    tid = std::string( 1, '0').append(std::to_string(thread));
  } else {
    tid = std::to_string(thread);
  }
  sprintf (buf,"\r(thd: %s) %s [%.*s>%*s] %d/%d", tid.c_str(), prefix.c_str(), lpad, PBSTR, rpad, "", icur, tot);
  std::string strouti = buf;
  std::cout << strouti;
  std::cout.flush();
  if(icur == tot-1) {
    sprintf (buf, "\r(thd: %s) %s [%.*s%*s] %d/%d", tid.c_str(), prefix.c_str(), PBWIDTH, PBSTR, 0, "", tot, tot);
    std::string stroutf  = buf;
    std::cout << stroutf;
    std::cout << " ";
  }
}


int main(int argc, char **argv) {

  int k1 = 0, k2 = 0;
  int nthd = 4, ctr = 0;
  int *sidx = new int[nthd]();
  int tot = 21;
  //int tot = 100;
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
    //printprogress_omp("nshots:", i - sidx[omp_get_thread_num()], csize, omp_get_thread_num());
    printprogress_strm_omp("nshots:", i - sidx[omp_get_thread_num()], csize, omp_get_thread_num());
    firstiter = false;
    //printprogress("nshots", i, 100);
    //printprogress_strm("nshots", i, 100);
  }
  //printf("\n");
  std::cout << std::endl;
  delete[] sidx;
}
