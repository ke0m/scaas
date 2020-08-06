/**
 * A complex tridiagonal solver.
 * Port of pctrist.f90 in SEPlib written by Paul Sava
 * @author: Joseph Jennings
 * @version: 2020.08.05
 */
#ifndef CTRIST_H_
#define CTRIST_H_

class ctrist {

  public:
    ctrist(int nd, float od, float dd, int nm, float om, float dm, float eps);

  private:
    int   *_idx;
    bool  *_flg;
    float *_wgt;
    float *_diag, *_offd;
};


#endif /* CTRIST_H_ */
