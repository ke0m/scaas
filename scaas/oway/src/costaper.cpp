#include <math.h>
#include <stdio.h>
#include "costaper.h"

void costaper(int dim, int dim1, int n1, int n2, int *n, int *nw, int *s, float *data) {

  /* Build the taper */
  int maxdim = 5;
  float *w[maxdim];
  for(int i = 0; i < dim; ++i) {
    if(nw[i] > 0) {
      /* Allocate memory for taper */
      w[i] = new float[nw[i]]();
      for(int iw = 0; iw < nw[i]; ++iw) {
        float wi = sinf(0.5*M_PI*(iw+1.)/(nw[i]+1.));
        w[i][iw] = wi*wi;
      }
    } else {
      w[i] = NULL;
    }
  }

  /* Apply taper */
  for(int i2 = 0; i2 < n2; ++i2) {
    /* Get a pointer to the data */
    float *x = data + i2*n1;
    for(int i = 0; i <= dim1; ++i) {
      if(nw[i] <= 0) continue;
      for(int j = 0; j < n1/n[i]; ++j) {
        int i0 = first_index(i, j, dim1+1, n, s);
        for(int iw = 0; iw < nw[i]; ++iw) {
          float wi = w[i][iw];
          x[i0+iw*s[i]]          *= wi;
          x[i0+(n[i]-1-iw)*s[i]] *= wi;
        }
      }
    }
  }

  /* Free memory */
  for(int idim = 0; idim < maxdim; ++idim) {
    if(nw[idim] > 0) delete[] w[idim];
  }
}

int first_index (int i          /* dimension [0...dim-1] */,
    int j        /* line coordinate */,
    int dim        /* number of dimensions */,
    const int *n /* box size [dim] */,
    const int *s /* step [dim] */)
/*< Find first index for multidimensional transforms >*/
{
  int i0, n123, ii;
  int k;

  n123 = 1;
  i0 = 0;
  for (k=0; k < dim; k++) {
    if (k == i) continue;
    ii = (j/n123)%n[k]; /* to cartesian */
    n123 *= n[k];
    i0 += ii*s[k];      /* back to line */
  }

  return i0;
}

