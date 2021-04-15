"""
Performs matrix-vector multiplication

@author: Joseph Jennings
@version: 2020.06.17
"""
import numpy as np
from opt.linopt.opr8tr import operator

class matmul(operator):
  """ Matrix-vector multiplication operator """

  def __init__(self,mat):
    """
    matmul constructor

    Parameters:
      mat - input matrix (2D numpy array)
    """
    if(len(mat.shape) != 2):
      raise Exception("Input matrix array must be 2D")

    self.__mat = mat
    self.__nrows = mat.shape[0]
    self.__ncols = mat.shape[1]

  def forward(self,add,mod,dat):
    """
    Multiplies the input matrix by the model

    Parameters:
      add - whether to add to the output (True/False)
      mod - input model vector
      dat - output data vector
    """
    if(mod.shape[0] != self.__ncols):
      raise Exception("Size of model must be size of cols of matrix")
    if(dat.shape[0] != self.__nrows):
      raise Exception("Size of data must be size of rows of matrix")

    if(not add):
      dat[:] = 0.0

    dat += np.dot(self.__mat,mod)

  def adjoint(self,add,mod,dat):
    """
    Multiplies the transpose of the matrix by the data

    Parameters:
      add - whether to add to the output (True/False)
      mod - output model vector
      dat - input data vector 
    """
    if(mod.shape[0] != self.__ncols):
      raise Exception("Size of model must be size of cols of matrix")
    if(dat.shape[0] != self.__nrows):
      raise Exception("Size of data must be size of rows of matrix")

    if(not add):
      mod[:] = 0.0

    mod += np.dot(self.__mat.T,dat)

  def dottest(self,add=False):
    """ Performs the dot product test of the operator """
    m  = np.random.rand(self.__ncols).astype('float32')
    mh = np.zeros(self.__ncols,dtype='float32')
    d  = np.random.rand(self.__nrows).astype('float32')
    dh = np.zeros(self.__nrows,dtype='float32')

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


