/**
 * A header-only progress bar for printing
 * the progress from serial and parallel loops
 * @author: Joseph Jennings
 * @version: 2020.06.27
 */

#ifndef PROGRESSBAR_H_
#define PROGRESSBAR_H_

#define PBSTR "============================================================="
#define PBWIDTH 60

#include <stdio.h>
#include <string>
#include <iostream>

inline void printprogress(std::string prefix, int icur, int tot) {
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

inline void printprogress_strm(std::string prefix, int icur, int tot) {
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

inline void printprogress_omp(std::string prefix, int icur, int tot, int thread) {
  double percentage = (double)icur/tot;
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad-1;
  std::string tid;
  if(thread < 10) {
    tid = std::string( 1, '0').append(std::to_string(thread));
  } else {
    tid = std::to_string(thread);
  }
  printf ("\r(thd: %s) %s [%.*s>%*s] %d/%d", tid.c_str(), prefix.c_str(), lpad, PBSTR, rpad, "", icur, tot);
  fflush (stdout);
  if(icur == tot-1) {
    printf ("\r(thd: %s) %s [%.*s%*s] %d/%d", tid.c_str(), prefix.c_str(), PBWIDTH, PBSTR, 0, "", tot, tot);
    printf (" ");
  }
}

inline void printprogress_strm_omp(std::string prefix, int icur, int tot, int thread) {
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

#endif /* PROGRESSBAR_H_ */
