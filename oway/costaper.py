"""
Applies a cosine taper at the borders
of an N-D cube
@author: Joseph Jennings
@version: 2020.07.17
"""
import numpy as np
from oway.costapkern import costapkern

def costaper(dat,nw1=0,nw2=0,nw3=0,nw4=0,nw5=0):
  """
  Applies a cosine taper at the borders of an 
  N-D cube

  Parameters:
    dat - an N-D numpy array
    nw1 - length of taper along fast axis
    nw2 - length of taper along second axis
    nw3 - length of taper along third axis
    nw4 - length of taper along fourth axis
    nw5 - length of taper along fifth axis

  Returns:
    the tapered cube
  """
  # First check that the input is of type float32
  if(dat.dtype != 'float32'):
    raise Exception("Input data array must be of type 'float32'")

  # Create the necessary inputs for the kernel function
  dim = len(dat.shape)
  ns  = np.ones(5,dtype='int32')
  nws = np.zeros(5,dtype='int32')
  s   = np.zeros(5,dtype='int32')
  nws[:] = np.asarray([nw1,nw2,nw3,nw4,nw5])

  dim1 = -1
  for i in range(dim):
    ns[i] = dat.shape[dim-i-1]
    if(nws[i] > 0): dim1 = i

  n1 = n2 = 1
  for i in range(dim):
    if(i <= dim1):
      s[i] = n1
      n1 *= ns[i]
    else:
      n2 *= ns[i]

  # Create a copy of the data
  tmp = np.copy(dat)
  # Apply taper
  costapkern(dim,dim1,n1,n2,ns,nws,s,tmp)

  return tmp

