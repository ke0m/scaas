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

class ssr3{
  public:
    ssr3(int nx, int ny, int nz, int nh, int nw, int nr,
         float dx, float dy, float dz, float dh, float dw,
         float ow, float eps, int ntx, int nty);
    void ssr3ssf_modonew(int iw, float *slo, float *ref, std::complex<float> *wav, std::complex<float> *dat);
    void ssr3ssf_modallw(float *slo, float *ref, std::complex<float> *wav, std::complex<float> *dat);
    void ssr3ssf_migonew();
    void ssr3ssf_migallw();
    void slow3(int nr);
    void build_taper(int nt1, int nt2);
    void apply_taper(std::complex<float> *slc);
    void apply_taper(std::complex<float> *slcin, std::complex<float> *slcot);
    ~ssr3() {
      delete[] _tapx; delete[] _tapy;
    }

  private:
    int _nx, _ny, _nz, _nh, _nw, _nr;
    int _ntx, _nty;
    int _onestp;
    float _dx, _dy, _dy, _dz, _dh, _dw;
    float _ow;
    float _eps;
    float *_tapx, *_tapy;
};

#endif /* SSR3_H_ */
