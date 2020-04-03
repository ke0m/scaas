"""
Three-dimensional triangular smoother
@author: Joseph Jennings
@version: 2020.03.18
"""
import numpy as np
from scaas.trismoothkern import smooth2

def smooth(data,rect1=1,rect2=1,rect3=1):
  """
  Applies a triangular smoother along specified
  dimensions

  Parameters:
    data  - The input data
    rect1 - Number of points to smooth along fast axis [-1, no smoothing]
    rect2 - Number of points to smooth along second axis [-1, no smoothing]
    rect3 - Number of points to smooth along slow axis [-1, no smoothing]
  """
  # Create the necessary inputs for the kernel function
  dim = len(data.shape)
  ns    = np.ones(3,dtype='int32')
  rects = np.ones(3,dtype='int32')
  s     = np.zeros(3,dtype='int32')
  rects[0] = rect1; rects[1] = rect2; rects[2] = rect3

  dim1 = -1
  for i in range(dim):
    ns[i] = data.shape[dim-i-1]
    if(rects[i] > 1): dim1 = i

  n1 = n2 = 1
  for i in range(dim):
    if(i <= dim1):
      s[i] = n1
      n1 *= ns[i]
    else:
      n2 *= ns[i]

  # Create copy of data
  tmp = np.copy(data)
  # Apply kernel
  smooth2(dim1,n1,n2,ns,rects,s,tmp)

  return tmp

