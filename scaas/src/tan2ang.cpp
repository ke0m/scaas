#include <math.h>
#include "tan2ang.h"

void tan2ang(int nz, int nta, float ota, float dta,
    int na, float oa, float da, int ext, float *tan, float *ang) {

  /* Temporary arrays */
  float *tmp = new float[nta]();
  float *spl = new float[nta+2*ext]();
  float w[4];
  /* Arrays for tridiagonal solver */
  float *d = new float[2*(nta+2*ext)]();
  float *o = new float[2*(nta+2*ext)]();
  float *x = new float[2*(nta+2*ext)]();
  /* Build the diagonal and off-diagonal of the toeplitz matrix */
  float diag = 2./3.; float odiag = 1./6.;
  build_toeplitz_matrix(nta+2*ext, diag, odiag, false, d, o);

  /* Loop over all depths */
  for (int iz = 0; iz < nz; iz++) {
    /* Grab a slice along z (interpolate tangents) */
    for (int ita = 0; ita < nta; ita++) {
      tmp[ita] = tan[ita*nz + iz];
    }
    /* fint1_set */
    extend1(ext,nta,tmp,spl);             // Pad tmp and save to spl
    spl[0] *= (5/6.);
    spl[nta+2*ext-1] *= (5/6.);           // Prepare endpoints for matrix solve
    solve_toeplitz_matrix(nta,d,o,x,spl); // Solve matrix and store in spl

    /* Loop over angle */
    for (int ia=0; ia < na; ia++) {
      float a = oa + ia*da;
      float n = tanf(a*(M_PI/180));

      float f = (n - ota) / dta;
      int fint = (int)f;

      if (fint >= 0 && fint < nta) {
        ang[ia*nz + iz] = interp(ext,w,spl,fint,f-fint);
      } else {
        ang[ia*nz + iz] = 0.;
      }
    }
  }

  /* Free memory */
  delete[] tmp;
  delete[] spl;
  delete[] d; delete[] o; delete[] x;
}

void build_toeplitz_matrix(int n,
    float diag  /* diagonal */,
    float offd  /* off-diagonal */,
    bool damp   /* damping */,
    float *d, float *o)
{

  d[0] = damp? diag+offd: diag;
  for (int k = 1; k < n; k++) {
    o[0*n + k-1] = offd / d[0*n + k-1];
    d[0*n + k]   = diag - offd * o[0*n + k-1];
  }
  if (damp) d[0*n + n-1] += offd;
  d[1*n + n-1] = damp? diag+offd: diag;
  for (int k = n-2; k >= 0; k--) {
    o[1*n + k] = offd / d[1*n + k+1];
    d[1*n + k] = diag - offd * o[1*n + k];
  }
  if (damp) d[1*n + 0] += offd;
}

void solve_toeplitz_matrix(int n,
    float *d, /* Precomputed diagonal */
    float *o, /* Precomputed offdiagonal */
    float *x, /* Temporary array */
    float* b /* in - right-hand side, out - solution */)
/*< invert the matrix >*/
{
  x[0] = b[0];
  for (int k = 1; k < n; k++) {
    x[0*n + k] = b[k] - o[0*n + k-1] * x[0*n + k-1];
  }
  x[1*n + n-1] = b[n-1];
  for (int k = n-2; k >= 0; k--) {
    x[1*n + k] = b[k] - o[1*n + k] * x[1*n + k+1];
  }
  b[n-1] = x[0*n + n-1] / d[0*n + n-1];
  for (int k = n-2; k >= n/2; k--) {
    b[k] = x[0*n + k] / d[0*n + k] - o[0*n + k] * b[k+1];
  }
  b[0] = x[1*n + 0] / d[1*n + 0];
  for (int k = 1; k < n/2; k++) {
    b[k] = x[1*n + k] / d[1*n + k] - o[1*n + k-1] * b[k-1];
  }
}

float interp(int ext, float *w, float *spl, int i, float x) {

  spline4_int(x,w);

  float f = 0;
  for(int j = 0; j < 4; ++j) {
    int k = i + ext/2 + j + 1;
    f += w[j]*spl[k];
  }

  return f;

}

void spline4_int(float x, float* w)
/*< Cubic spline interpolation >*/
{
  float x2 = x*x;
  w[0] = (1. + x*((3. - x)*x-3.))/6.;
  w[1] = (4. + 3.*(x -2.)*x2)/6.;
  w[2] = (1. + 3.*x*(1. + (1. - x)*x))/6.;
  w[3] = x2*x/6.;
}

void extend1 (int ne     /* padding */,
    int nd     /* data length */,
    float *dat /* data [nd] */,
    float *ext /* extension [nd+2*ne] */)
/*< 1-D extension >*/
{
  int nw = 3;
  const float a[] = {7./3., -5./3., 1./3.};

  for (int i=0; i < nd; i++) {
    ext[ne+i] = dat[i];
  }
  for (int i=ne-1; i >= 0; i--) {
    float s = 0.;
    for (int j=0; j < nw; j++) {
      s += a[j]*ext[i+j+1];
    }
    ext[i] = s;
  }
  for (int i=nd+ne; i < nd+2*ne; i++) {
    float s = 0.;
    for (int j=0; j < nw; j++) {
      s += a[j]*ext[i-j-1];
    }
    ext[i] = s;
  }
}

