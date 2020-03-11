from opt.linopt.opr8tr import operator

class identity(operator):
  """ Identity operator """

  def forward(self,add,mod,dat):
    """ Sets the data equal to the model """
    if(not add):
      dat[:] = 0.0
    dat +=  mod

  def adjoint(self,add,mod,dat):
    """ Sets the model equal to the data """
    if(not add):
      mod[:] = 0.0
    mod += dat

