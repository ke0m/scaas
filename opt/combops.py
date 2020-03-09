"""
Combination operators

Types of combinations:
  Chain operator    [AB]

  Row operator      [A B]

  Column operator   [A]
                    [B]

  Diagonal operator [A 0]
                    [0 B]
@author: Joseph Jennings
@version: 2020.03.08
"""
from opt.opr8tr import operator
import numpy as np

class chainop(operator):
  """ A chain operator """

  def __init__(self,ops,dims):
    """
    chainop constructor

    Parameters:
      ops - a list of operators to be chained
      dims - a list of dictionaries that contain the 
             dimensions of the inputs and outputs of the arrays
             For example dims = [{'nrows': 10, 'ncols': 10},...]

    Note that the operators will be applied in the order
    they come in the list:

    [A,B,C,D,E] = EDCBAm
    """
    self.__ops = ops
    self.__nops = len(ops)
    for iop in range(self.__nops):
      if(not isinstance(self.__ops[iop],operator)):
        raise Exception("Elements of ops list must be of type operator")

    # Check if the dimensions are valid
    self.__mdim = dims[ 0]['ncols']
    self.__ddim = dims[-1]['nrows']
    for idim in range(len(dims)-1):
      if(np.prod(dims[idim]['nrows']) != np.prod(dims[idim+1]['ncols'])):
        print("Operator %d has %d rows and operator %d has %d cols"%(idim,
          np.prod(dims[idim]['nrows']),idim+1,np.prod(dims[idim+1]['ncols'])))
        raise Exception("Dimensions of ops don't match")

    # Save dimensions
    self.__dims = dims

  def forward(self,add,mod,dat):
    """
    Applies the forward of a generic chain operator

    Parameters:
      add - whether to add to the total output
      mod - input model vector
      dat - output data vector
    """
    # Check if model and data dimensions match
    if(mod.shape != self.__mdim):
      raise Exception("Input model does not match first operator in chain")
    if(dat.shape != self.__ddim):
      raise Exception("Output data does not match last operator in chain")

    # Apply first to model
    tmp1 = mod
    for iop in range(self.__nops):
      if(iop < self.__nops-1):
        # Allocate temporary array
        tmp2 = np.zeros(self.__dims[iop]['nrows'],dtype='float32')
        self.__ops[iop].forward(False,tmp1,tmp2)
        # The current output becomes the next input
        tmp1 = tmp2
      else:
        self.__ops[iop].forward(add,tmp1,dat)

  def adjoint(self,add,mod,dat):
    """
    Applies the adjoint of a generic chain operator

    Parameters:
      add - whether to add to the total output
      mod - output model vector
      dat - input data vector
    """
    # Check if model and data dimensions match
    if(mod.shape != self.__mdim):
      raise Exception("Input model does not match first operator in chain")
    if(dat.shape != self.__ddim):
      raise Exception("Output data does not match last operator in chain")

    # Apply first to data
    tmp1 = dat
    for iop in range(self.__nops-1,-1,-1):
      if(iop > 0):
        # Allocate temporary array
        tmp2 = np.zeros(self.__dims[iop]['ncols'],dtype='float32')
        self.__ops[iop].adjoint(False,tmp2,tmp1)
        # The current output becomes the next input
        tmp1 = tmp2
      else:
        self.__ops[iop].adjoint(add,mod,tmp1)

