"""
A 1D linear interpolation operator 
written in numba jit

@author: Joseph Jennings
@version: 2020.06.16
"""
from opt.linopt.opr8tr import operator
import numpy as np
from numba import jit, int32, float32

class lint(operator):
  """ A linear interpolation operator """

  def __init__(self,nm,om,dm,crd):
    """
    lint constructor

    Parameters:
      nm  - size of model (regularly sampled)
      om  - origin of model axis
      dm  - sampling of model axis
      nd  - size of data  (irregularly sampled)
      crd - input coordinates of the values (of the irregularly sampled data)
    """
    self.__nm  = nm; self.__om = om; self.__dm = dm
    self.__nd  = len(crd)
    self.__crd = crd

    if(self.__crd.shape[0] != self.__nd):
      raise Exception("coordinate array must be same length as data")

  def forward(self,add,mod,dat):
    """
    Applies the forward linear interpolation operator

    Parameters:
      add - whether to add to the input (True/False)
      mod - input model numpy array
      dat - input data numpy array
    """
    if(mod.shape[0] != self.__nm or dat.shape[0] != self.__nd):
      raise Exception("lint forward: input shapes do not match those passed to constructor")

    if(add == False):
      dat[:] = 0.0

    forward_lint(self.__om,self.__dm,self.__nm,self.__nd,self.__crd,mod,dat)

  def adjoint(self,add,mod,dat):
    """
    Applies the adjoint transient convolution operator

    Parameters
      add - whether to add to the input (True/False)
      mod - output model numpy array
      dat - input data numpy array
    """
    if(mod.shape[0] != self.__nm or dat.shape[0] != self.__nd):
      raise Exception("lint adjoint: input shapes do not match those passed to constructor")

    if(add == False):
      mod[:] = 0.0

    adjoint_lint(self.__om,self.__dm,self.__nm,self.__nd,self.__crd,mod,dat)

  def dottest(self,add=False):
    """ Performs the dot product test of the operator """
    # Create random model and data
    m  = np.random.rand(self.__nm).astype('float32')
    mh = np.zeros(self.__nm,dtype='float32')
    d  = np.random.rand(self.__nd).astype('float32')
    dh = np.zeros(self.__nd,dtype='float32')

    if(add):
      self.forward(True,m ,dh)
      self.adjoint(True,mh,d )
      dotm = np.dot(m,mh); dotd = np.dot(d,dh)
      print("Dot product test (add==True):")
      print("Dotm = %f Dotd = %f"%(dotm,dotd))
      print("Absolute error = %f"%(abs(dotm-dotd)))
      print("Relative error = %f"%(abs(dotm-dotd)/dotd))
    else:
      self.forward(False,m ,dh)
      self.adjoint(False,mh,d )
      dotm = np.dot(m,mh); dotd = np.dot(d,dh)
      print("Dot product test (add==False):")
      print("Dotm = %f Dotd = %f"%(dotm,dotd))
      print("Absolute error = %f"%(abs(dotm-dotd)))
      print("Relative error = %f"%(abs(dotm-dotd)/dotd))

@jit(nopython=True)
def forward_lint(om,dm,nm,nd,crds,modl,data):
  for ic in range(nd):
    f = (crds[ic] - om)/dm
    i = int(f + 0.5)
    fx = f - i; gx = 1 - fx
    if(i >= 0 and i < nm-1):
      data[ic] += gx*modl[i] + fx*modl[i+1]

@jit(nopython=True)
def adjoint_lint(om,dm,nm,nd,crds,modl,data):
  for ic in range(nd):
    f = (crds[ic] - om)/dm
    i = int(f + 0.5)
    fx = f - i; gx = 1 - fx
    if(i >= 0 and i < nm-1):
      modl[i]   += gx*data[ic]
      modl[i+1] += fx*data[ic]

