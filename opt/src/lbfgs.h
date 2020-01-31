#ifndef LBFGS_H
#define LBFGS_H

#include <cstdlib>
#include <cstdio>

#define epsilon 1e-5f
#define xtol 1e-16f
#define ftol 1e-4f
#define gtol 0.9f
#define stpmin 1e-10f
#define stpmax 1e10f
#define maxfev 20
#define xtrapf 4.f

#define nisave 8
#define ndsave 14

float sdot(size_t n,float *x,float *y);

void saxpy(size_t n,float a,float *x,float *y);

void saxpyz(size_t n,float a,float *x,float *y,float *z);

void mcstep(float &stx,float &fx,float &dx,float &sty,float &fy,float &dy,float &stp,float &fp,float &dp,bool &brackt,float &stmin,float &stmax,int &info);

void mcsrch(size_t n,float *x,float &f,float *g,float *s,float &stp,int &info,int &nfev,float *wa,int *isave,float *dsave);

void lbfgs(size_t n,int &m,float *x,float &f,float *g,bool diagco,float *diag,float *w,int &iflag,int *isave,float *dsave);

extern "C"{
    void lbfgs_(int *N,int *M,float *X,float *F,float *G,int *DIAGCO,float *DIAG,int IPRINT[],float *EPS,float *XTOL,float *W,int *IFLAG);  
}

struct{
    int LP,MP;
    float GTOL,STPMIN,STPMAX;
} LB2;

void steepest(size_t n,float *x,float &f,float *g,float *diag,float *w,int &iflag,int *isave,float *dsave);

void nlcg(size_t n,float *x,float &f,float *g,float *diag,float *w,int &iflag,int *isave,float *dsave);

#endif
