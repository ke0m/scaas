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

    # Check consistency between dims and ops
    if(self.__nops != len(dims)):
      raise Exception("Number of dimensions (%d) must equal number of operators (%d)"
          %(len(dims),self.__nops))

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

  def totaldims(self):
    """ Return the total dimensions of the chain operator """
    totaldims = {}
    totaldims['ncols'] = self.__mdim
    totaldims['nrows'] = self.__ddim

    return totaldims

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

class rowop(operator):
  """ Row operator """

  def __init__(self,ops,dims):
    """
    rowop constructor

    Parameters:
      ops  - a list of operators used to form the row operator
      dims - a list of dictionaries that contain the 
             dimensions of the inputs and outputs of the arrays
             For example dims = [{'nrows': 10, 'ncols': 10},...]

    Note that the row operator will be formed in the order in which the
    operators are supplied. For example
    ops = [A,B,C,...] will result in  [A,B,C,...][ma]
                                                 [mb]
                                                 [mc]
                                                 [::]
    """
    self.__ops = ops 
    self.__nops = len(ops)
    for iop in range(self.__nops):
      if(not isinstance(self.__ops[iop],operator)):
        raise Exception("Elements of ops list must be of type operator")

    # Check consistency between dims and ops
    if(self.__nops != len(dims)):
      raise Exception("Number of dimensions (%d) must equal number of operators (%d)"
          %(len(dims),self.__nops))

    # Check if the dimensions are valid
    self.__ddim = dims[-1]['nrows']
    for idim in range(len(dims)-1):
      if(np.prod(dims[idim]['nrows']) != np.prod(dims[idim+1]['nrows'])):
        print("Operator %d has %d rows and operator %d has %d rows"%(idim,
          np.prod(dims[idim]['nrows']),idim+1,np.prod(dims[idim+1]['nrows'])))
        raise Exception("Dimensions of ops don't match")

    # Save dimensions
    self.__dims = dims

  def totaldims(self):
    """ Returns the total dims of the row operator """
    totaldims = {}
    totaldims['nrows'] = self.__ddim
    totaldims['ncols'] = 0
    for iop in range(self.__nops):
      totaldims['ncols']  += dims[iop]['ncols']

    return totaldims

  def forward(self,add,mod,dat):
    """
    Applies the forward of a generic row operator

    Parameters:
      add - whether to add to the total output
      mod - a list of input models [mc,mb,ma]
      dat - output data vector
    """
    # Check if model is a list
    if(not isinstance(mod,list)):
      raise Exception("Input model must be a list of numpy arrays")
    if(len(mod) != self.__nops):
      raise Exception("Number of models in list must be same as number of operators")
    # Check if data dimensions match
    if(dat.shape != self.__ddim):
      raise Exception("Output data does not match row operator")

    iadd = add
    # Apply each operator and add to the output
    for iop in range(self.__nops):
      self.__ops[iop].forward(iadd,mod[iop],dat)
      iadd = True

  def adjoint(self,add,mod,dat):
    """
    Applies the adjoint of a generic row operator (column operator)

    Parameters:
      add - whether to add to the total output
      mod - a list of output models [mc,mb,ma]
      dat - input data vector
    """
    # Check if model is a list
    if(not isinstance(mod,list)):
      raise Exception("Output model must be a list of numpy arrays")
    if(len(mod) != self.__nops):
      raise Exception("Number of models in list must be same as number of operators")
    # Check if data dimensions match
    if(dat.shape != self.__ddim):
      raise Exception("Output data does not match row operator")

    # Each adjoint gives a different model
    for iop in range(self.__nops):
      self.__ops[iop].forward(add,m[iop],data)

class colop(operator):
  """ Column operator """

  def __init__(self,ops,dims):
    """
    colop constructor

    Parameters:
      ops  - a list of operators used to form the column operator
      dims - a list of dictionaries that contain the 
             dimensions of the inputs and outputs of the arrays
             For example dims = [{'nrows': 10, 'ncols': 10},...]

    Note that the column operator will be formed in the order in which the
    operators are supplied. For example
    ops = [A,B,C,...] will result in  [A]m
                                      [B]
                                      [C]
                                      [:]
    """
    self.__ops = ops 
    self.__nops = len(ops)
    for iop in range(self.__nops):
      if(not isinstance(self.__ops[iop],operator)):
        raise Exception("Elements of ops list must be of type operator")

    # Check consistency between dims and ops
    if(self.__nops != len(dims)):
      raise Exception("Number of dimensions (%d) must equal number of operators (%d)"%(len(dims),self.__nops))

    # Check if the dimensions are valid
    self.__ddim = dims[-1]['nrows']
    for idim in range(len(dims)-1):
      if(np.prod(dims[idim]['cols']) != np.prod(dims[idim+1]['ncols'])):
        print("Operator %d has %d cols and operator %d has %d cols"%(idim,
          np.prod(dims[idim]['ncols']),idim+1,np.prod(dims[idim+1]['ncols'])))
        raise Exception("Dimensions of ops don't match")

    # Save dimensions
    self.__dims = dims

  def totaldims(self):
    """ Returns the total dims of the column operator """
    totaldims = {}
    totaldims['ncols'] = self.__mdim
    totaldims['nrows'] = 0
    for iop in range(self.__nops):
      totaldims['nrows']  += dims[iop]['nrows']

    return totaldims

  def forward(self,add,mod,dat):
    """
    Applies the forward of a generic column operator

    Parameters:
      add - whether to add to the total output
      mod - input model vector
      dat - output list of data vectors
    """
    # Check if data is a list
    if(not isinstance(dat,list)):
      raise Exception("Output data must be a list of numpy arrays")
    if(len(dat) != self.__nops):
      raise Exception("Number of data vectors in list must be same as number of operators")
    # Check if data dimensions match
    if(mod.shape != self.__mdim):
      raise Exception("Input model does not match column operator")

    # Each forward gives a different data
    for iop in range(self.__nops):
      self.__ops[iop].forward(add,m,data[iop])

  def adjoint(self,add,mod,dat):
    """
    Applies the adjoint of a generic column operator (row operator)

    Parameters:
     add - whether to add to the total output
     mod - output model vector
     dat - input list of data vectors
    """
    # Check if data is a list
    if(not isinstance(dat,list)):
      raise Exception("Input data must be a list of numpy arrays")
    if(len(dat) != self.__nops):
      raise Exception("Number of data vectors in list must be same as number of operators")
    # Check if data dimensions match
    if(mod.shape != self.__mdim):
      raise Exception("Output model does not match column operator")

    iadd = add
    # Apply each operator and add to the output
    for iop in range(self.__nops):
      self.__ops[iop].adjoint(iadd,mod,dat[iop])
      iadd = True

class diagop(operator):
  """ A diagonal operator """

  def __init__(self,ops,dims):
    """
    diagop constructor
    
    Parameters
      ops  - a list of operators used to form the row operator
      dims - a list of dictionaries that contain the 
             dimensions of the inputs and outputs of the arrays
             For example dims = [{'nrows': 10, 'ncols': 10},...]

    Note that the diagonal operator will be formed in the order in which
    operators are supplied. For example if ops = [A,B,C,...] then
    [A 0 0][ma] will be formed
    [0 B 0][mb]
    [0 0 C][mc]
    """
    self.__ops = ops 
    self.__nops = len(ops)
    for iop in range(self.__nops):
      if(not isinstance(self.__ops[iop],operator)):
        raise Exception("Elements of ops list must be of type operator")

    # Check consistency between dims and ops
    if(self.__nops != len(dims)):
      raise Exception("Number of dimensions (%d) must equal number of operators (%d)"%
          (len(dims),self.__nops))

  def totaldims(self):
    """ Returns the total dims of the column operator """
    totaldims = {}
    totaldims['ncols'] = 0
    totaldims['nrows'] = 0
    for iop in range(self.__nops):
      totaldims['nrows'] += dims[iop]['nrows']
      totaldims['ncols'] += dims[iop]['ncols'] 
      
    return totaldims

  def forward(self,add,mod,dat):
    """
    Applies the forward of a generic diagonal operator

    Parameters:
      add - whether to add to the total output
      mod - input list of model vectors
      dat - output list of data vectors
    """
    # Check if model and data are lists
    if(not isinstance(dat,list) or not isinstance(mod,list)):
      raise Exception("dat and mod must be a list of numpy arrays")

    # Apply each operator to each model giving the data
    for iop in range(nops):
      self.__ops[iop].forward(add,mod[iop],dat[iop])

  def adjoint(self,add,mod,dat):
    """
    Applies the adjoint of a generic diagonal operator

    Parameters:
      add - whether to add to the total output
      mod - output list of model vectors
      dat - output list of data vectors
    """
    # Check if model and data are lists
    if(not isinstance(dat,list) or not isinstance(mod,list)):
      raise Exception("dat and mod must be a list of numpy arrays")

    # Apply each operator to each model giving the data
    for iop in range(nops):
      self.__ops[iop].adjoint(add,mod[iop],dat[iop])

