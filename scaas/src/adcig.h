/**
 * Functions for converting subsurface offset to angle
 * @author: Joseph Jennings
 * @version: 2020.03.22
 */
#ifndef ADCIG_H_
#define ADCIG_H_

void convert2ang(int nx, int nh, float oh, float dh,
    int nta, float ota, float dta, int na, float oa, float da, int nz, float oz, float dz,
    int ext, float *off, float *ang, int nthrd, bool verb);

#endif /* ADCIG_H_ */
