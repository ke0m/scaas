"""
Four-dimensional triangular smoother
@author: Joseph Jennings
@version: 2020.06.16
"""
import numpy as np
from scaas.trismoothkern import smooth2
from opt.linopt.opr8tr import operator

def smooth(data,rect1=1,rect2=1,rect3=1,rect4=1):
  """
  Applies a triangular smoother along specified
  dimensions

  Parameters:
    data  - The input data
    rect1 - Number of points to smooth along fast axis [-1, no smoothing]
    rect2 - Number of points to smooth along second axis [-1, no smoothing]
    rect3 - Number of points to smooth along third axis [-1, no smoothing]
    rect4 - Number of points to smooth along slow axis [-1, no smoothing]
  """
  # First check that the input is of type float32
  if(data.dtype != np.zeros(1,dtype='float32').dtype):
    raise Exception("Input data array must be of type 'float32'")

  # Create the necessary inputs for the kernel function
  dim = len(data.shape)
  ns    = np.ones(4,dtype='int32')
  rects = np.ones(4,dtype='int32')
  s     = np.zeros(4,dtype='int32')
  rects[0] = rect1; rects[1] = rect2; rects[2] = rect3; rects[3] = rect4

  dim1 = -1
  for i in range(dim):
    ns[i] = data.shape[dim-i-1]
    if(rects[i] > -1): dim1 = i

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

class smoothop(operator):
  """ A triangle smoothing operator """

  def __init__(self,nsin,rect1=1,rect2=1,rect3=1,rect4=1):
    """
    smoothop constructor
    
    Parameters:
      nsin  - shape of input signal/image (arr.shape)
      rect1 - Number of points to smooth along fast axis [-1, no smoothing]
      rect2 - Number of points to smooth along second axis [-1, no smoothing]
      rect3 - Number of points to smooth along third axis [-1, no smoothing]
      rect4 - Number of points to smooth along slow axis [-1, no smoothing]
    """
    # Create the necessary inputs for the kernel function
    dim = len(nsin)
    self.__nsin  = nsin
    self.__ns    = np.ones(4,dtype='int32')
    self.__rects = np.ones(4,dtype='int32')
    self.__s     = np.zeros(4,dtype='int32')
    self.__rects[0] = rect1; self.__rects[1] = rect2; 
    self.__rects[2] = rect3; self.__rects[3] = rect4
    
    self.__dim1 = -1
    for i in range(dim):
      self.__ns[i] = nsin[dim-i-1]
      if(self.__rects[i] > -1): self.__dim1 = i 

    self.__n1 = self.__n2 = 1 
    for i in range(dim):
      if(i <= self.__dim1):
        self.__s[i] = self.__n1
        self.__n1  *= self.__ns[i]
      else:
        self.__n2  *= self.__ns[i]

  def forward(self,add,mod,dat):
    """
    Applies the foward triangle smoothing operator

    Parameters:
      add - Whether to add to the output [True/False]
      mod - input unsmooth signal/image (numpy array)
      dat - output smoothed signal/image (numpy/array)
    """
    # Check data size
    if(mod.shape != dat.shape):
      raise Exception("trismoothop: model must have same shape as data")
    if(list(mod.shape) != self.__nsin):
      raise Exception("trismoothop: shape of input must match shape passed to constructor")

    if(not add):
      dat[:] = 0.0

    # Create copy of data
    tmp = np.copy(mod)
    # Apply kernel
    smooth2(self.__dim1,self.__n1,self.__n2,self.__ns,self.__rects,self.__s,tmp)
    dat[:] += tmp

  def adjoint(self,add,mod,dat):
    """
    Applies the foward triangle smoothing operator

    Parameters:
      add - Whether to add to the output [True/False]
      mod - input unsmooth signal/image (numpy array)
      dat - output smoothed signal/image (numpy/array)
    """
    # Check data size
    if(mod.shape != dat.shape):
      raise Exception("trismoothop: model must have same shape as data")
    if(list(mod.shape) != self.__nsin):
      raise Exception("trismoothop: shape of input must match shape passed to constructor")

    if(not add):
      mod[:] = 0.0

    # Create copy of data
    tmp = np.copy(dat)
    # Apply kernel
    smooth2(self.__dim1,self.__n1,self.__n2,self.__ns,self.__rects,self.__s,tmp)
    mod[:] += tmp

  def dottest(self,add=False):
    """ Performs the dot product test of the operator """
    # Create the model and the data
    m  = np.random.rand(*self.__nsin).astype('float32')
    mh = np.zeros(self.__nsin,dtype='float32')
    d  = np.random.rand(*self.__nsin).astype('float32')
    dh = np.zeros(self.__nsin,dtype='float32')

    if(add):
      self.forward(True,m ,dh)
      self.adjoint(True,mh,d )
      dotm = np.dot(m.flatten(),mh.flatten()); 
      dotd = np.dot(d.flatten(),dh.flatten())
      print("Dot product test (add==True):")
      print("Dotm = %f Dotd = %f"%(dotm,dotd))
      print("Absolute error = %f"%(abs(dotm-dotd)))
      print("Relative error = %f"%(abs(dotm-dotd)/dotd))
    else:
      self.forward(False,m ,dh)
      self.adjoint(False,mh,d )
      dotm = np.dot(m.flatten(),mh.flatten()); 
      dotd = np.dot(d.flatten(),dh.flatten())
      print("Dot product test (add==False):")
      print("Dotm = %f Dotd = %f"%(dotm,dotd))
      print("Absolute error = %f"%(abs(dotm-dotd)))
      print("Relative error = %f"%(abs(dotm-dotd)/dotd))

