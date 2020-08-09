/**
 * A complex tridiagonal solver.
 * A port of pctrist.f90 in SEPlib written by Paul Sava
 * @author: Joseph Jennings
 * @version: 2020.08.05
 */
#ifndef CTRIST_H_
#define CTRIST_H_

#include <complex.h>

class ctrist {

  public:
    ctrist(int n, float o, float d, float eps);
    void define(float *coord);
    void apply(std::complex<float> *model, std::complex<float> *data);
    void solve(std::complex<float> *diag, std::complex<float> *offd, std::complex<float> *model);
    ~ctrist(){
      delete[] _idx; delete[] _flg;
      delete[] _wgt; delete[] _diag; delete[] _offd;
      delete[] _dd;  delete[] _oo;
    }

  private:
    int _nm;
    float _om, _dm, _eps;
    int   *_idx;
    bool  *_flg;
    std::complex<float> *_wgt;
    std::complex<float> *_diag, *_offd;
    std::complex<float> *_dd, *_oo;
};


#endif /* CTRIST_H_ */
