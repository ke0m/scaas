/**
 * A header-only progress bar for printing
 * the progress from serial and parallel loops
 * @author: Joseph Jennings
 * @version: 2020.03.23
 */

#ifndef PROGRESSBAR_H_
#define PROGRESSBAR_H_

#define PBSTR "============================================================="
#define PBWIDTH 60

#include <stdio.h>
#include <string>

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

inline void printprogress_omp(std::string prefix, int icur, int tot, int thread) {
  double percentage = (double)icur/tot;
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad-1;
  std::string tid;
  if(thread < 10) {
    tid = std::string( 1, '0').append(std::to_string(thread));

  }
  printf ("\r(thd: %s) %s [%.*s>%*s] %d/%d", tid.c_str(), prefix.c_str(), lpad, PBSTR, rpad, "", icur, tot);
  fflush (stdout);
  if(icur == tot-1) {
    printf ("\r(thd: %s) %s [%.*s%*s] %d/%d", tid.c_str(), prefix.c_str(), PBWIDTH, PBSTR, 0, "", tot, tot);
    printf(" ");
  }
}

#endif /* PROGRESSBAR_H_ */
