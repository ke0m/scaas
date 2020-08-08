/**
 * Functions for converting from offset to angle.
 * Works in the Kz-Khy-Khx domain via the radial
 * trace transform
 * @author: Joseph Jennings
 * @version: 2020.08.06
 */

#ifndef SCAAS_SRC_ADCIGKZKX_H_
#define SCAAS_SRC_ADCIGFKZKX_H_

#include <complex>
#include "ctrist.h"

void convert2angkzkykx(int ngat,
                       int nz, float oz, float dz,
                       int nhy, float ohy, float dhy,
                       int nhx, float ohx, float dhx,
                       float oa,  float da,
                       std::complex<float> *off, std::complex<float> *ang,
                       float eps, int nthrd, bool verb);

void convertone2angkhx(int nz, int nkhx, float okhx, float dkhx, int nkhy,
                       float* stretch, float eps, ctrist *solv,
                       std::complex<float> *off, std::complex<float> *ang);

void forwardshift(int nz, float oz, float dz,
                  int nhy, float ohy, float dhy,
                  int nhx, float ohx, float dhx,
                  std::complex<float>* data);

void inverseshift(int nz, float oz, float dz,
                  int nhy, int nhx, std::complex<float> *data);

#endif /* SCAAS_SRC_ADCIGKZKX_H_ */
