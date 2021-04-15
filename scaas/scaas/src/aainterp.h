/**
 * Performs an anti-aliased linear interpolation
 * A port of aainterp.c as part of Madagascar
 * written by Sergey Fomel
 * @author: Joseph Jennings
 * @version: 2020.03.19
 */

#ifndef AAINTERP_H_
#define AAINTERP_H_

class aainterp {
  public:
    aainterp();
    aainterp(int n1, float o1, float d1, int n2);
    void define(const float *coord, const float *delt, const float *amp,
        float *x, bool *m, float *w, float *a);
    void forward(bool add, int n1, int n2, const float *x, const bool *m, const float *w, const float *a,
        float *ord, float *modl);
    void adjoint(bool add, int n1, int n2, const float *x, const bool *m, const float *w, const float *a,
        float *ord, float *modl);
    void doubint(bool dble, int n, float *trace);

  private:
    int _n1, _n2, _nk;
    float _o1, _d1;
};

#endif /* AAINTERP_H_ */
