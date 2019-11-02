#include"laplacianADJISPC.h"

export void laplacianADJISPC(uniform int nx, uniform int nz, uniform float idx2, uniform float idz2, 
    uniform float mod[], uniform float dat[]) {

  uniform float coef0 = -2.9272222519;
  uniform float coef1 = 1.6666666665;
  uniform float coef2 = -0.2380952381;
  uniform float coef3 = 0.0396825397;
  uniform float coef4 = -0.00496031745;
  uniform float coef5 = 0.0003174603;

  for(uniform int ix = 5; ix < nx-5; ++ix) {
    foreach(iz=5...nz-5){
      mod[ix*nz + iz] = coef0*dat[ix*nz + iz] * (idx2 + idz2) + 
        coef1*(dat[(ix  )*nz + iz+1] + dat[(ix  )*nz + iz-1])*idz2 +
        coef2*(dat[(ix  )*nz + iz+2] + dat[(ix  )*nz + iz-2])*idz2 +
        coef3*(dat[(ix  )*nz + iz+3] + dat[(ix  )*nz + iz-3])*idz2 +
        coef4*(dat[(ix  )*nz + iz+4] + dat[(ix  )*nz + iz-4])*idz2 +
        coef5*(dat[(ix  )*nz + iz+5] + dat[(ix  )*nz + iz-5])*idz2 +
        coef1*(dat[(ix+1)*nz + iz  ] + dat[(ix-1)*nz + iz  ])*idx2 +
        coef2*(dat[(ix+2)*nz + iz  ] + dat[(ix-2)*nz + iz  ])*idx2 +
        coef3*(dat[(ix+3)*nz + iz  ] + dat[(ix-3)*nz + iz  ])*idx2 +
        coef4*(dat[(ix+4)*nz + iz  ] + dat[(ix-4)*nz + iz  ])*idx2 +
        coef5*(dat[(ix+5)*nz + iz  ] + dat[(ix-5)*nz + iz  ])*idx2;
    }
  }
}
