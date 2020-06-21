"""
Transient convolution operator with
numba/just-in-time compiler

@author: Joseph Jennings
@version: 2020.06.16
"""
from opt.linopt.opr8tr import operator
import numpy as np
from numba import jit, int32, float32

class tcai(operator):
  """ A transient convolution operator """

  def __init__(self,nm,nd,flt):
    """
    tcai constructor

    Parameters:
      nm  - size of model
      nd  - size of data
      flt - input filter coefficients (numpy array)
    """
    self.__nm  = nm
    self.__nd  = nd
    self.__nf  = flt.shape[0]
    self.__flt = flt

    if(self.__nf != self.__nd - self.__nm + 1):
      raise Exception("tcai: Size of filter does not match output and input")

  def forward(self,add,mod,dat):
    """
    Applies the forward transient convolution operator

    Parameters:
      add - whether to add to the input (True/False)
      mod - input model numpy array
      dat - input data numpy array
    """
    if(mod.shape[0] != self.__nm or dat.shape[0] != self.__nd):
      raise Exception("tcai forward: input shapes do not match those passed to constructor")

    if(add == False):
      dat[:] = 0.0

    forward_tcai(self.__nf,self.__nm,self.__nd,self.__flt,mod,dat)

  def adjoint(self,add,mod,dat):
    """
    Applies the adjoint transient convolution operator

    Parameters
      add - whether to add to the input (True/False)
      mod - output model numpy array
      dat - input data numpy array
    """
    if(mod.shape[0] != self.__nm or dat.shape[0] != self.__nd):
      raise Exception("tcai forward: input shapes do not match those passed to constructor")

    if(add == False):
      mod[:] = 0.0

    adjoint_tcai(self.__nf,self.__nm,self.__nd,self.__flt,mod,dat)

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
def forward_tcai(nf,nm,nd,flt,modl,data):
  for i in range(nf):
    for im in range(nm):
      j = im + i - 1 
      data[j] += modl[im]*flt[i]

@jit(nopython=True)
def adjoint_tcai(nf,nm,nd,flt,modl,data):
  for i in range(nf):
    for im in range(nm):
      j = im + i - 1 
      modl[im] += data[j]*flt[i]

