/**
 * Functions for applying a triangular smoothing
 * operator.
 * A port of triangle.c from Madagascar written by Sergey Fomel
 * @author: Joseph Jennings
 * @version: 2020.03.18
 */
#ifndef TRISMOOTH_H_
#define TRISMOOTH_H_

// Main smoothing functions
void smooth2(int dim1, int n1, int n2, int *n, int *rect, int *s, float *data);
// Utility functions
void fold (int o, int d, int nx, int nb, int np, const float *x,       float *tmp);
void fold2(int o, int d, int nx, int nb, int np,       float *x, const float *tmp);
void doubint( int nx, float *xx, bool der);
void doubint2(int nx, float *xx, bool der);
void triple  (int o, int d, int nx, int nb,       float *x, const float *tmp, float wt);
void dtriple (int o, int d, int nx, int nb,       float *x, const float *tmp, float wt);
void triple2 (int o, int d, int nx, int nb, const float *x,       float *tmp, float wt);
void dtriple2(int o, int d, int nx, int nb, const float *x,       float *tmp, float wt);
int first_index(int i, int j, int dim, const int *n, const int *s);
void saxpy(int n, float a, const float *x, int sx, float *y, int sy);

#endif /* TRISMOOTH_H_ */
