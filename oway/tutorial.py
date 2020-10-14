"""
Functions for a one-way wave equation tutorial

@author: Joseph Jennings
@version: 2020.10.13
"""

def build_taper(ntx,nty) -> numpy.ndarray:
  """
  Builds a 2D tapering function

  Parameters:
    ntx - the size of the taper in the x direction
    nty - the size of the taper in the y direction

  Returns the taper function along x and the taper
  function along y
  """
  pass

def build_karray(dx,dy,bx,by):
  """
  Builds the wavenumber array that is used in the
  single square root operator

  Parameters:
    dx - x sampling interval
    dy - y sampling interval
    bx - size of padded array in x
    by - size of padded array in y

  Returns the wavenumber array
  """
  pass

def nrefs(nrmax):
  pass

def quantile():
  pass
