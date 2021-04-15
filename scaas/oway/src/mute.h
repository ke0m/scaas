/**
 * Applies a mute to seismic gathers
 * A port of Mmutter.c in Madagscar (written by Sergey Fomel)
 * which is originally an extension of Mutter.f written by Jon Claerbout
 * @author: Joseph Jennings
 * @version: 2020.07.11
 */

#ifndef MUTE_H_
#define MUTE_H_

void muteone(int nt, float ot, float dt, bool abs0, bool inner, bool hyper,
             float tp, float slope0, float slopep, float x,
             float *datin, float *datot);

void muteall(int n3, int n2, float o2, float d2, int n1, float o1, float d1,
             float tp, float t0, float v0, float slope0, float slopep, float x0,
             bool abs, bool inner, bool hyper, bool half,
             float *datin, float *datot);


#endif /* MUTE_H_ */
