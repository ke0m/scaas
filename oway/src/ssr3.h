/**
 * One-way wave equation solver via the
 * single square root operator.
 * Based on the ssr3.c functions from Paul Sava in Madagascar
 *
 * @author: Joseph Jennings
 * @version: 2020.07.22
 */

#ifndef SSR3_H_
#define SSR3_H_

#include <complex>
#include <fftw3.h>

class ssr3{
  public:
    ssr3(int nx,   int ny,   int nz,
         float dx, float dy, float dz,
         int nw, float ow, float dw, float eps,
         int ntx, int nty, int px, int py,
         float dtmax, int nrmax, int nthrds);
    void set_slows(float *slo);
    /* Modeling */
    void inject_src (int nsrc, float *srcy, float *srcx, float oy, float ox,
                       std::complex<float> *wav, std::complex<float> *sou);
    void inject_srct(int nsrc, float *srcy, float *srcx, float oy, float ox,
                       std::complex<float> *wav, std::complex<float> *sou);
    void ssr3ssf_modonew(int iw, float *ref, std::complex<float> *wav, std::complex<float> *dat, int ithrd);
    void ssr3ssf_modallw(float *ref, std::complex<float> *wav, std::complex<float> *dat, bool verb);
    /* Modeling and imaging helper functions */
    void restrict_data (int nrec, float *recy, float *recx, float oy, float ox,
                        std::complex<float> *dat, std::complex<float> *rec);
    void restrict_datat(int nrec, float *recy, float *recx, float oy, float ox,
                        std::complex<float> *dat, std::complex<float> *rec);
    void inject_data (int nrec, float *recy, float *recx, float oy, float ox,
                     std::complex<float> *rec, std::complex<float> *dat);
    void inject_datat(int nrec, float *recy, float *recx, float oy, float ox,
                     std::complex<float> *rec, std::complex<float> *dat);
    /* Conventional imaging */
    void ssr3ssf_migonew(int iw, std::complex<float> *dat, std::complex<float> *wav, float *img, int ithrd);
    void ssr3ssf_migallw(std::complex<float> *dat, std::complex<float> *wav, float *img, bool verb);
    /* Extended imaging */
    void set_ext(int nhy, int nhx, bool sym);
    void del_ext();
    void ssr3ssf_migoffonew(int iw, std::complex<float> *dat, std::complex<float> *wav,
                            int bly, int ely, int blx, int elx, float *img, int ithrd);
    void ssr3ssf_migoffallw(std::complex<float> *dat, std::complex<float> *wav, float *img, bool verb);
    /* Model zero offset */
    void ssr3ssf_modallwzo(float *img, std::complex<float> *dat, bool verb);
    void ssr3ssf_modonewzo(int iw, float *img, std::complex<float> *dat, int ithrd);
    /* Migrate zero offset */
    void ssr3ssf_migallwzo(std::complex<float> *dat, float *img, bool verb);
    void ssr3ssf_migonewzo(int iw, std::complex<float> *dat, float *img, int ithrd);
    /* Wavefield zero offset */
    void ssr3ssf_fwfallwzo(std::complex<float> *dat, std::complex<float> *wfl, bool verb);
    void ssr3ssf_fwfonewzo(int iw, std::complex<float> *dat, std::complex<float> *wfl, int ithrd);
    void ssr3ssf_awfallwzo(std::complex<float> *dat, std::complex<float> *wfl, bool verb);
    void ssr3ssf_awfonewzo(int iw, std::complex<float> *dat, std::complex<float> *wfl, int ithrd);
    void ssr3ssf(std::complex<float> w, int iz, float *scur, float *snex, std::complex<float> *slccur, std::complex<float> *slcnex, int ithrd);
    void ssr3ssf(std::complex<float> w, int iz, float *scur, float *snex, std::complex<float> *slc, int ithrd);
    void build_refs(int nz, int nrmax, int ns, float dsmax, float *slo, int *nr, float *sloref);
    int nrefs(int nrmax, float dsmax, int ns, float *slo, float *sloref);
    void build_taper(int nt1, int nt2, float *tapx, float *tapy);
    void apply_taper(std::complex<float> *slc);
    void apply_taper(std::complex<float> *slcin, std::complex<float> *slcot);
    void build_karray(float dx, float dy, int bx, int by, float *kk);
    float quantile(int q, int n, float *a);
    ~ssr3() {
      delete[] _nr; delete[] _sloref;
      delete[] _tapx; delete[] _tapy; delete[] _kk;
      /* Remove FFTW plans */
      for(int ithrd = 0; ithrd < _nthrds; ++ithrd) {
        fftwf_destroy_plan(_fplans[ithrd]); fftwf_destroy_plan(_iplans[ithrd]);
        delete[] _wxks[ithrd]; delete[] _wkks[ithrd]; delete[] _wxxs[ithrd];
      }
      delete[] _fplans; delete[] _iplans;
      delete[] _wxks;   delete[] _wkks; delete[] _wxxs;
    }

  private:
    int _nx, _ny, _nz, _nw;
    int _ntx, _nty, _px, _py, _bx, _by;
    int _nthrds;
    int _nrmax, *_nr;
    int _rnhy, _rnhx, _blx, _elx, _bly, _ely;
    float _dx, _dy, _dz, _dw, _dsmax, _dsmax2;
    float _ow;
    float _eps;
    float *_slo, *_sloref;
    float *_tapx, *_tapy;
    float *_kk;
    std::complex<float> **_wxks, **_wkks, **_wxxs;
    fftwf_plan *_fplans, *_iplans;
    float **_imgar;
};

/* Helper functions */
void interp_slow(int nz, int nvy, float ovy, float dvy, int nvx, float ovx, float dvx,
                 int ny, float oy, float dy, int nx, float ox, float dx,
                 float *sloin, float *sloot);

#endif /* SSR3_H_ */
