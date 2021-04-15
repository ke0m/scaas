"""
A hyperbolic radon operator

@author: Joseph Jennings
@version: 2020.07.12
"""
from opt.linopt.opr8tr import operator
import numpy as np
from numba import jit, int32, float32

class slow(operator):
  """
  2D slowness operator that maps spikes in slowness and depth
  to hyperbolas in time and space
  """

  def __init__(self,nq,oq,dq,nz,oz,dz,nx,ox,dx,nt,ot,dt):
    """
    slow constructor

    Parameters:
      nq - number of slownesses
      oq - slowness origin
      dq - slowness sampling
      nz - number of depths
      oz - depth origin
      dz - depth sampling
      nx - number of offsets
      ox - origin of offsets
      dx - offset sampling
      nt - number of time samples
      ot - origin of time axis
      dt - temporal sampling
    """
    # Get model dimensions
    self.__nq = nq; self.__oq = oq; self.__dq = dq
    self.__nz = nz; self.__oz = oz; self.__dz = dz
    # Get data dimensions
    self.__nx = nx; self.__ox = ox; self.__dx = dx
    self.__nt = nt; self.__ot = ot; self.__dt = dt

  def get_dat_size(self):
    return self.__nx,self.__nt

  def get_mod_size(self):
    return self.__nq,self.__nz

  def forward(self,add,mod,dat):
    """
    Applies the forward operator:
    Spikes in depth-slowness to hyperbolas in time-space

    Parameters:
      add - whether to add to the input [True/False]
      mod - slowness model (s,z)
      dat - hyperbolas (t,x)
    """
    if(mod.shape != (self.__nq,self.__nz)):
      raise Exception("Model must have same size as parameters passed to constructor")
    if(dat.shape != (self.__nx,self.__nt)):
      raise Exception("Data must have same size as parameters passed to constructor")

    # Zero the data if add == false
    if(add == False):
      dat[:] = 0.0

    forward2D_1(self.__oq,self.__dq,
                self.__oz,self.__dz,
                self.__ox,self.__dx,
                self.__ot,self.__dt,
                mod,dat)

  def adjoint(self,add,mod,dat):
    """
    Applies the adjoint operator:
    Hyperbolas in time-space to spikes in depth-slowness

    Parameters:
      add - add to the output [True/False]
      mod - slowness model (s,z)
      dat - hyperbolas (t,x)
    """
    if(mod.shape != (self.__nq,self.__nz)):
      raise Exception("Model must have same size as parameters passed to constructor")
    if(dat.shape != (self.__nx,self.__nt)):
      raise Exception("Data must have same size as parameters passed to constructor")

    if(add == False):
      mod[:] = 0.0

    adjoint2D_1(self.__oq,self.__dq,
                self.__oz,self.__dz,
                self.__ox,self.__dx,
                self.__ot,self.__dt,
                mod,dat)

  def dottest(self,add=False):
    """ Performs the dot product test of the operator """
    # Create random model and data
    m  = np.random.rand(self.__nq,self.__nz).astype('float32')
    mh = np.zeros(m.shape,dtype='float32')
    d  = np.random.rand(self.__nx,self.__nt).astype('float32')
    dh = np.zeros(d.shape,dtype='float32')

    if(add):
      self.forward(True,m ,dh)
      self.adjoint(True,mh,d )
      dotm = np.dot(m.flatten(),mh.flatten()) 
      dotd = np.dot(d.flatten(),dh.flatten())
      print("Dot product test (add==True):")
      print("Dotm = %f Dotd = %f"%(dotm,dotd))
      print("Absolute error = %f"%(abs(dotm-dotd)))
      print("Relative error = %f"%(abs(dotm-dotd)/dotd))
    else:
      self.forward(False,m ,dh)
      self.adjoint(False,mh,d )
      dotm = np.dot(m.flatten(),mh.flatten()) 
      dotd = np.dot(d.flatten(),dh.flatten())
      print("Dot product test (add==False):")
      print("Dotm = %f Dotd = %f"%(dotm,dotd))
      print("Absolute error = %f"%(abs(dotm-dotd)))
      print("Relative error = %f"%(abs(dotm-dotd)/dotd))

@jit(nopython=True)
def forward2D_1(oq,dq,oz,dz,ox,dx,ot,dt,modl,data):
  # Model dimensions
  nq = modl.shape[0]; nz = modl.shape[1]
  # Data dimensions
  nx = data.shape[0]; nt = data.shape[1]
  for iq in range(nq):
    q = oq + iq*dq
    for ix in range(nx):
      x = ox + ix*dx
      sx = np.abs(q * x)
      for iz in range(nz):
        z = oz + iz*dz
        # Given slowness, offset and depth, compute time from NMO equation
        t = np.sqrt( z * z + sx * sx )
        # Compute linear interpolation weights
        f = (t - oz)/dz
        it = int(f + 0.5)
        fx = f - it; gx = 1.0 - fx
        # Linearly interpolate
        if(it >=0 and it < nt-1):
          data[ix][it+0] += gx*modl[iq][iz]
          data[ix][it+1] += fx*modl[iq][iz]

@jit(nopython=True)
def adjoint2D_1(oq,dq,oz,dz,ox,dx,ot,dt,modl,data):
  # Model dimensions
  nq = modl.shape[0]; nz = modl.shape[1]
  # Data dimensions
  nx = data.shape[0];  nt = data.shape[1]
  for iq in range(nq):
    q = oq + iq*dq
    for ix in range(nx):
      x = ox + ix*dx
      sx = np.abs(q * x)
      for iz in range(nz):
        z = oz + iz*dz
        # Given slowness, offset and depth, compute time from NMO equation
        t = np.sqrt( z * z + sx * sx )
        # Compute linear interpolation weights
        f = (t - oz)/dz
        it = int(f + 0.5)
        fx = f - it; gx = 1.0 - fx
        # Linearly interpolate
        if(it >=0 and it < nt - 1):
          modl[iq][iz] += gx*data[ix][it] + fx*data[ix][it+1]

