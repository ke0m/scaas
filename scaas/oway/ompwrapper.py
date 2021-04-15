import numpy as np
from oway.ompfunc import ompfunc

def ompwrap(ina,scale,ntry,nthrds):
  n = ina.shape[0]
  ota = np.zeros(n,dtype='float32')
  for i in range(ntry):
    ompfunc(n,scale,ina,ota,nthrds)

  return ota

def ompwrap2(n,scale,ntry,nthrds):
  ina = np.ones(n,dtype='float32')
  ota = np.zeros(n,dtype='float32')
  for i in range(ntry):
    ompfunc(n,scale,ina,ota,nthrds)

  return ota

def ompwrap3(ina,scale,ntry,nthrds):
  n = ina.shape[0]
  ota = np.zeros(n,dtype='float32')
  for i in range(ntry):
    ompfuncpy(n,scale,ina,ota,nthrds)

  return ota

def ompfuncpy(n,scale,ina,ota,nthrds):
  ota[:] = ina[:]*scale

