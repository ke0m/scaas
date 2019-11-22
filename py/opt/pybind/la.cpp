#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "lbfgs.h"

float sdot(size_t n,float *x,float *y){
    float temp=0.0;
    if(n<=0) return temp;
    #pragma omp parallel for reduction(+:temp) num_threads(16)
    for(size_t i=0;i<n;i++) temp+=x[i]*y[i];
    return temp;
}

void saxpy(size_t n,float a,float *x,float *y){
    if(n<=0 || a==0.0f) return;
    #pragma omp parallel for num_threads(16)
    for(size_t i=0;i<n;i++) y[i]+=a*x[i];
    return;
}

void saxpyz(size_t n,float a,float *x,float *y,float *z){
    if(n<=0) return;
    if(a==0.0f){
        memcpy(z,y,n*sizeof(float));
        return;
    }
    #pragma omp parallel for num_threads(16)
    for(size_t i=0;i<n;i++) z[i]=a*x[i]+y[i];
    return;
}

