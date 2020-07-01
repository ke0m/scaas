/**
 * One-way wave equation solver via the
 * single square root operator.
 * Based on the ssr3.c functions from Paul Sava in Madagascar
 *
 * @author: Joseph Jennings
 * @version: 2020.06.21
 */

#ifndef SSR3_H_
#define SSR3_H_

#include <complex>
#include "kiss_fft.h"

class ssr3{
  public:
    ssr3(int nx,   int ny,   int nz,   int nh,
         float dx, float dy, float dz, float dh,
         int nw, float ow, float dw, float eps,
         int ntx, int nty, int px, int py,
         float dtmax, int nrmax);
    void set_slows(float *slo);
    void ssr3ssf_modonew(int iw, float *ref, std::complex<float> *wav, std::complex<float> *dat);
    void ssr3ssf_modallw(float *ref, std::complex<float> *wav, std::complex<float> *dat);
    void ssr3ssf_migonew();
    void ssr3ssf_migallw();
    void ssr3ssf(std::complex<float> w, int iz, float *scur, float *snex, std::complex<float> *slccur, std::complex<float> *slcnex);
    void ssr3ssf(std::complex<float> w, int iz, float *scur, float *snex, std::complex<float> *slc);
    void build_refs(int nz, int nrmax, int ns, float dsmax, float *slo, int *nr, float *sloref);
    int nrefs(int nrmax, float dsmax, int ns, float *slo, float *sloref);
    void build_taper(int nt1, int nt2, float *tapx, float *tapy);
    void apply_taper(std::complex<float> *slc);
    void apply_taper(std::complex<float> *slcin, std::complex<float> *slcot);
    void build_karray(float dx, float dy, int bx, int by, float *kk);
    float quantile(int q, int n, float *a);
    void fft2(bool inv, kiss_fft_cpx *pp);
    kiss_fft_cpx cmul(kiss_fft_cpx a, float b);
    ~ssr3() {
      delete[] _nr; delete[] _sloref;
      delete[] _tapx; delete[] _tapy; delete[] _kk;
      free(_fwd1); free(_inv1); free(_fwd2); free(_inv2);
    }

  private:
    int _nx, _ny, _nz, _nh, _nw;
    int _ntx, _nty, _px, _py, _bx, _by;
    int _onestp;
    int _nrmax, *_nr;
    float _dx, _dy, _dz, _dh, _dw, _dsmax, _dsmax2;
    float _ow;
    float _eps;
    float *_slo, *_sloref;
    float *_tapx, *_tapy;
    float *_kk;
    kiss_fft_cfg _fwd1, _inv1, _fwd2, _inv2;

};

#endif /* SSR3_H_ */
