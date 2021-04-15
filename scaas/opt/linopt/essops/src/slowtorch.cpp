/**
 * A PyTorch/Pybind interface for the slowness operator
 * @author: Joseph Jennings
 * @version: 2020.07.13
 */

#include <torch/extension.h>
#include "slow.h"

void slowforward_torch(int nq, float oq, float dq, int nz, float oz, float dz,
                       int nx, float ox, float dx, int nt, float ot, float dt,
                       at::Tensor mod, at::Tensor dat) {

  slowforward(nq, oq, dq, nz, oz, dz,
              nx, ox, dx, nt, ot, dt,
              mod.data<float>(), dat.data<float>());
}


void slowadjoint_torch(int nq, float oq, float dq, int nz, float oz, float dz,
                       int nx, float ox, float dx, int nt, float ot, float dt,
                       at::Tensor mod, at::Tensor dat) {

  slowadjoint(nq, oq, dq, nz, oz, dz,
              nx, ox, dx, nt, ot, dt,
              mod.data<float>(), dat.data<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("slowforward", &slowforward_torch, "Forward slowness transform");
  m.def("slowbackward", &slowadjoint_torch, "Adjoint slowness transform");
}

