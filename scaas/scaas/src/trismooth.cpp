#include "trismooth.h"

void smooth2(int dim1, int n1, int n2, int *n, int *rect, int *s, float *data){
  for(int i2 = 0; i2 < n2; i2++) {
    /* Get a pointer to the data */
    float *x = data + i2*n1;
    for(int i = 0; i <= dim1; ++i) {
      if(rect[i] <= 1) continue;
      /* Filter parameters for the current dimension */
      float wt = 1/(static_cast<float>(rect[i])*static_cast<float>(rect[i]));
      int np = n[i] + 2*rect[i];
      /* Temporary array */
      float *tmp = new float[np]();
      for(int j = 0; j < n1/n[i]; ++j) {
        int i0 = first_index(i,j,dim1+1,n,s);
        /* Smoothing */
        triple2(i0,s[i],n[i],rect[i],x,tmp,wt);
        doubint2(np,tmp,false);
        fold2(i0,s[i],n[i],rect[i],np,x,tmp);
      }
      /* Free memory */
      delete[] tmp;
    }
  }
}

void fold (int o, int d, int nx, int nb, int np, const float *x, float* tmp) {

  /* copy middle */
  for (int i=0; i < nx; i++)
    tmp[i+nb] = x[o+i*d];

  /* reflections from the right side */
  for (int j=nb+nx; j < np; j += nx) {
    for (int i=0; i < nx && i < np-j; i++)
      tmp[j+i] = x[o+(nx-1-i)*d];
    j += nx;
    for (int i=0; i < nx && i < np-j; i++)
      tmp[j+i] = x[o+i*d];
  }

  /* reflections from the left side */
  for (int j=nb; j >= 0; j -= nx) {
    for (int i=0; i < nx && i < j; i++)
      tmp[j-1-i] = x[o+i*d];
    j -= nx;
    for (int i=0; i < nx && i < j; i++)
      tmp[j-1-i] = x[o+(nx-1-i)*d];
  }

}

void fold2(int o, int d, int nx, int nb, int np, float *x, const float *tmp) {
  /* copy middle */
  for (int i=0; i < nx; i++)
    x[o+i*d] = tmp[i+nb];

  /* reflections from the right side */
  for (int j=nb+nx; j < np; j += nx) {
    for (int i=0; i < nx && i < np-j; i++)
      x[o+(nx-1-i)*d] += tmp[j+i];
    j += nx;
    for (int i=0; i < nx && i < np-j; i++)
      x[o+i*d] += tmp[j+i];
  }

  /* reflections from the left side */
  for (int j=nb; j >= 0; j -= nx) {
    for (int i=0; i < nx && i < j; i++)
      x[o+i*d] += tmp[j-1-i];
    j -= nx;
    for (int i=0; i < nx && i < j; i++)
      x[o+(nx-1-i)*d] += tmp[j-1-i];
  }

}

void doubint(int nx, float *xx, bool der) {

  /* integrate backward */
  float t = 0.;
  for (int i=nx-1; i >= 0; i--) {
    t += xx[i];
    xx[i] = t;
  }

  if (der) return;

  /* integrate forward */
  t=0.;
  for (int i=0; i < nx; i++) {
    t += xx[i];
    xx[i] = t;
  }

}

void doubint2 (int nx, float *xx, bool der) {

  /* integrate forward */
  float t=0.;
  for (int i=0; i < nx; i++) {
    t += xx[i];
    xx[i] = t;
  }

  if (der) return;

  /* integrate backward */
  t = 0.;
  for (int i=nx-1; i >= 0; i--) {
    t += xx[i];
    xx[i] = t;
  }
}

void triple (int o, int d, int nx, int nb, float* x, const float* tmp, float wt) {

  const float *tmp1 = tmp + nb;
  const float *tmp2 = tmp + 2*nb;

  for (int i=0; i < nx; i++) {
    x[o+i*d] = (2.*tmp1[i] - tmp[i] - tmp2[i])*wt;
  }
}

void dtriple (int o, int d, int nx, int nb, float* x, const float* tmp, float wt) {

  const float *tmp2 = tmp + 2*nb;

  for (int i=0; i < nx; i++) {
    x[o+i*d] = (tmp[i] - tmp2[i])*wt;
  }
}

void triple2 (int o, int d, int nx, int nb, const float* x, float* tmp, float wt) {

  for (int i=0; i < nx + 2*nb; i++) tmp[i] = 0;

  saxpy(nx,  -wt,x+o,d,tmp     ,1);
  saxpy(nx,2.*wt,x+o,d,tmp+nb  ,1);
  saxpy(nx,  -wt,x+o,d,tmp+2*nb,1);
}

void dtriple2 (int o, int d, int nx, int nb, const float* x, float* tmp, float wt) {

  for (int i=0; i < nx + 2*nb; i++) tmp[i] = 0;

  saxpy(nx,  wt,x+o,d,tmp     ,1);
  saxpy(nx, -wt,x+o,d,tmp+2*nb,1);
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

void saxpy(int n, float a, const float *x, int sx, float *y, int sy) {

  for(int i = 0; i < n; ++i) {
    int ix = i*sx; int iy = i*sy;
    y[iy] += a * x[ix];
  }
}
