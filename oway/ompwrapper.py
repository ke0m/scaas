import numpy as np
from oway.ompfunc import ompfunc

def ompwrap(scale,ina,nthrds):
  n = ina.shape[0]
  ota = np.zeros(n,dtype='float32')
  ompfunc(n,scale,ina,ota,nthrds)

  return ota

