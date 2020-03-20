/**
 * Performs a slant stack operation
 * A port of slant.c in Madagascar
 * Written originally by Sergey Fomel
 * @author: Joseph Jennings
 * @version: 2020.03.19
 */
#ifndef SLANTSTK_H_
#define SLANTSTK_H_


class slantstk {
  public:
    slantstk(bool rho, int nx, float ox, float dx,
        int ns, float os, float ds, int nt, float ot, float dt, float s11, float anti1);
    void forward(bool add, int nm, int nd, float *mod, float *dat);
    void adjoint(bool add, int nm, int nd, float *mod, float *dat);

  private:
    bool _rho;
    int _nx, _ns, _nt;
    float _ox, _os, _ot;
    float _dx, _ds, _dt;
    float _s11, _anti1;
};


#endif /* SLANTSTK_H_ */
