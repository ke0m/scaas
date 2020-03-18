/**
 * Half-order integration and differentation as
 * described by Claerbout in BEI.
 * A port of the halfint code in Madagascar written
 * by Sergey Fomel
 * @author: Joseph Jennings
 * @version: 2020.03.18
 */
#ifndef HALFINT_H_
#define HALFINT_H_

#include "kiss_fft.h"
#include "kiss_fftr.h"

class halfint {
  public:
    halfint(bool inv, int n1, float rho);
    void forward(bool add, int n1, float *mod, float *dat);
    void adjoint(bool add, int n1, float *mod, float *dat);
    ~halfint() {
      free(_forw); free(_invs);
      delete[] _cf;
    }

  private:
    /* Private members */
    int _n, _nw;
    kiss_fft_cpx *_cf;
    kiss_fftr_cfg _forw, _invs;
    /* Private functions */
    kiss_fft_cpx csqrtf(kiss_fft_cpx c);
    kiss_fft_cpx cdiv(kiss_fft_cpx a, kiss_fft_cpx b);
    kiss_fft_cpx cmul(kiss_fft_cpx a, kiss_fft_cpx b);
    kiss_fft_cpx conjf(kiss_fft_cpx z);
};



#endif /* HALFINT_H_ */
