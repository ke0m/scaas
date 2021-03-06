"""
Weights vector by a supplied weighting operator

@author: Joseph Jennings
@version: 2020.06.20
"""
import numpy as np
from opt.linopt.opr8tr import operator

class weight(operator):
  """ Weight operator """

  def __init__(self,wgt):
    """
    weight constructor

    Parameters:
      wgt - input matrix (ND numpy array)
    """
    self.__wgt = wgt
    self.__wshape = wgt.shape

  def forward(self,add,mod,dat):
    """ 
    Applies the weight matrix to the model

    Parameters:
      add - whether to add to the output (True/False)
      mod - input model vector
      dat - output data vector
    """
    if(dat.shape != mod.shape):
      raise Exception("data and model must have same shape")
    if(dat.shape != self.__wshape):
      raise Exception("data and weight operator must have same shape")

    if(not add):
      dat[:] = 0.0

    dat[:] += self.__wgt*mod

  def adjoint(self,add,mod,dat):
    """
    Applies the weight matrix to the data

    Parameters:
      add - whether to add to the output (True/False)
      mod - input model vector
      dat - output data vector
    """
    if(dat.shape != mod.shape):
      raise Exception("data and model must have same shape")
    if(mod.shape != self.__wshape):
      raise Exception("model and weight operator must have same shape")

    if(not add):
      mod[:] = 0.0

    mod[:] += self.__wgt*dat

  def dottest(self,add=False):
    m  = np.random.rand(*self.__wshape).astype('float32')
    mh = np.zeros(self.__wshape,dtype='float32')
    d  = np.random.rand(*self.__wshape).astype('float32')
    dh = np.zeros(self.__wshape,dtype='float32')

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

