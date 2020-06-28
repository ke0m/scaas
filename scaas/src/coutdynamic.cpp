#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include "progressbar.h"

#define PBSTR "============================================================="
#define PBWIDTH 60

int main() {
  char mystr[PBWIDTH + 10];
  sprintf(mystr,"\r%.*s",20,PBSTR);
  std::string cppstr1 = mystr;
  std::cout << cppstr1;
  std::cout.flush();
  usleep(1000000);
  sprintf(mystr,"\r%.*s",30,PBSTR);
  std::string cppstr2 = mystr;
  std::cout << cppstr2;
  //printf("\r%.*s",20,PBSTR);
  //fflush (stdout);
  //usleep(1000000);
  //printf("\r%.*s",30,PBSTR);
  //printprogress("test",0,10);
  //usleep(1000000);
  //printprogress("test",1,10);
}
