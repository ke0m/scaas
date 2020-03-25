/**
 * Converts the tangent of the aperture angle
 * to angle (necessary for offset to angle conversion)
 * A port of tan2ang within Madagascar
 * written by Sergey Fomel
 * @author: Joseph Jennings
 * @version: 2020.03.21
 */
#ifndef TAN2ANG_H_
#define TAN2ANG_H_

void tan2ang(int nz, int nta, float ota, float dta,
    int na, float oa, float da, int ext, float *tan, float *ang);

void build_toeplitz_matrix(int n, float diag, float offd, bool damp, float *d, float *o);

void solve_toeplitz_matrix(int n, float *d, float *o, float *x, float* b);

void spline4_int(float x, float* w);

float interp(int ext, float *w, float *spl, int i, float x);

void extend1 (int ne, int nd, float *dat, float *ext);


#endif /* TAN2ANG_H_ */
