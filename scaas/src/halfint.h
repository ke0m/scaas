/**
 * Half-order integration and differentation as
 * described by Claerbout in BEI.
 * A port of the halfint code in Madagascar written
 * by Sergey Fomel
 * @version: 2020.03.17
 * @author: Joseph Jennings
 */
#ifndef HALFINT_H_
#define HALFINT_H_

#include "kiss_fft.h"
#include "kiss_fftr.h"

class halfint {
  public:
    halfint(bool inv, int n1, float rho);
    ~halfint() {
      free(_forw); free(_invs);
      delete[] _cx; delete[] _cf;
    }

  private:
    /* Private members */
    int _n, _nw;
    kiss_fft_cpx *_cx, *_cf;
    kiss_fftr_cfg _forw, _invs;
    /* Private functions */
    kiss_fft_cpx csqrtf(kiss_fft_cpx c);
    kiss_fft_cpx cdiv(kiss_fft_cpx a, kiss_fft_cpx b);

};



#endif /* HALFINT_H_ */
