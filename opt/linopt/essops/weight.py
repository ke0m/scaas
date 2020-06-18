"""
Weights vector by a supplied weighting operator

@author: Joseph Jennings
@version: 2020.06.17
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
    self.__wmat = np.diag(wgt.flatten())
    self.__n = self.__wmat.shape[0]

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
    # Flatten the vectors
    modf = mod.flatten(); datf = dat.flatten()
    if(modf.shape[0] != self.__n):
      raise Exception("model shape does not match weight matrix")

    if(not add):
      datf[:] = 0.0

    datf += np.dot(self.__wmat,modf)

    dat[:] = datf.reshape(dat.shape)

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
    # Flatten the vectors
    modf = mod.flatten(); datf = dat.flatten()
    if(datf.shape[0] != self.__n):
      raise Exception("data shape does not match weight matrix")

    if(not add):
      modf[:] = 0.0

    modf += np.dot(self.__wmat.T,datf)

    mod[:] = modf.reshape(mod.shape)

  def dottest(self,add=False):
    m  = np.random.rand(self.__n).astype('float32')
    mh = np.zeros(self.__n,dtype='float32')
    d  = np.random.rand(self.__n).astype('float32')
    dh = np.zeros(self.__n,dtype='float32')

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

