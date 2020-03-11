from abc import ABC, abstractmethod 

class operator(ABC):
  """ An operator class for linear inversion """

  def forward(self,add,mod,dat):
    """ Applies the forward of the linear operator """
    pass

  def adjoint(self,add,mod,dat):
    """ Applies the adjoint of the linear operator """
    pass
