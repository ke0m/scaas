#include"laplacian10.h"

export void laplacian10(uniform int nx, uniform int nz, uniform float idx2, uniform float idz2, 
    uniform float mod[], uniform float dat[]) {

  uniform float coef0 = -2.9272222519;
  uniform float coef1 = 1.6666666665;
  uniform float coef2 = -0.2380952381;
  uniform float coef3 = 0.0396825397;
  uniform float coef4 = -0.00496031745;
  uniform float coef5 = 0.0003174603;

  for(uniform int iz = 5; iz < nz-5; ++iz) {
    foreach(ix=5...nx-5){
      dat[iz*nx + ix] = coef0*mod[iz*nx + ix] * (idx2 + idz2) + 
        coef1*(mod[(iz  )*nx + ix+1] + mod[(iz  )*nx + ix-1])*idx2 +
        coef2*(mod[(iz  )*nx + ix+2] + mod[(iz  )*nx + ix-2])*idx2 +
        coef3*(mod[(iz  )*nx + ix+3] + mod[(iz  )*nx + ix-3])*idx2 +
        coef4*(mod[(iz  )*nx + ix+4] + mod[(iz  )*nx + ix-4])*idx2 +
        coef5*(mod[(iz  )*nx + ix+5] + mod[(iz  )*nx + ix-5])*idx2 +
        coef1*(mod[(iz+1)*nx + ix  ] + mod[(iz-1)*nx + ix  ])*idz2 +
        coef2*(mod[(iz+2)*nx + ix  ] + mod[(iz-2)*nx + ix  ])*idz2 +
        coef3*(mod[(iz+3)*nx + ix  ] + mod[(iz-3)*nx + ix  ])*idz2 +
        coef4*(mod[(iz+4)*nx + ix  ] + mod[(iz-4)*nx + ix  ])*idz2 +
        coef5*(mod[(iz+5)*nx + ix  ] + mod[(iz-5)*nx + ix  ])*idz2;
    }
  }
}
