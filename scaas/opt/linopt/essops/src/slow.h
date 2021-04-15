/**
 * Slowness transform
 * @author: Joseph Jennings
 * @version: 2020.07.13
 */
#ifndef SLOW_H_
#define SLOW_H_

void slowforward(int nq, float oq, float dq, int nz, float oz, float dz,
                 int nx, float ox, float dx, int nt, float ot, float dt,
                 float *mod, float *dat);

void slowadjoint(int nq, float oq, float dq, int nz, float oz, float dz,
                 int nx, float ox, float dx, int nt, float ot, float dt,
                 float *mod, float *dat);

#endif /* SLOW_H_ */
