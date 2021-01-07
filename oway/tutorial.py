"""
Functions for a one-way wave equation tutorial

@author: Joseph Jennings
@version: 2020.10.13
"""
import numpy as np
from genutils.ptyprint import progressbar

class ssr3tut:

  def __init__(self,nx,ny,nz,
               dx,dy,dz,
               nw,ow,dw,eps=0,
               ntx=0,nty=0,px=0,py=0,
               dtmax=5e-05,nrmax=3):
    """
    Constructor for SSR3 tutorial class

    Parameters:
      nx    - number of x samples
      ny    - number of y samples
      nz    - number of z samples
      nw    - number of frequencies
      ow    - frequency origin
      dw    - frequency sampling
      eps   - stability parameter [0.0]
      dtmax - maximum time error [5e-05]
      nrmax - maximum number of reference velocities
    """
    # Spatial axes
    self.__nz,self.__ny,self.__nx = nx,ny,nz
    self.__dz,self.__dy,self.__dx = dx,dy,dz
    self.__bx = nx + px
    self.__by = ny + py
    # Build the taper
    self.__tapx,self.__tapy = self.build_taper(ntx,nty)
    # Build spatial frequencies
    self.__kk = build_karray(dx,dy,bx,by)

    # Slowness array
    self.__slo = None

    # Reference velocities
    self.__nr     = np.zeros(nz,dype='int32')
    self.__nrmax  = nrmax
    self.__dsmax  = dtmax/dz
    self.__sloref = np.zeros([nz,nrmax],dtype='float32')

  def set_slows(self, slo):
    """
    Sets the slowness array and builds the reference
    slowness

    Parameters:
      slo - slowness array [nz,ny,nx]

    Sets internally the slowness array and computes the reference
    velocities
    """
    # Save slowness array
    self.__slo = slo

    # Compute reference slowness with depth
    for iz in range(nz):
      self.__nr[iz] = nrefs(self.__nrmax, self.__dsmax, self.__nx*self.__ny, 
                            slo[iz], self.__sloref[iz])

    # Build reference slownesses
    for iz in range(nz-1):
      self.__sloref[iz] = 0.5*(self.__sloref[iz] + self.__sloref[iz+1])

  def mod_allw(self, ref, wav, verb=False):
    """
    Linearized modeling of all frequencies using a SSR SSF operator 

    Parameters:
      ref  - the reflectivity model [nz,ny,nx]
      wav  - the input wavelet (FT'ed in time) [nw,ny,nx]
      verb - verbosity flag [False]
    """
    if(self.__slo is None): 
      raise Exception("Must run set_slows before modeling or migration")

    for iw in progressbar(range(self.__nw),"nw:",verb=verb):
      dat[iw] = self.ssr3ssf_modonew(iw, ref, wav[iw])

  def mod_onew(self, iw, ref, wav):
    """
    """
    pass

  def build_taper(self,ntx,nty) -> numpy.ndarray:
    """
    Builds a 2D tapering function
  
    Parameters:
      ntx - the size of the taper in the x direction
      nty - the size of the taper in the y direction
  
    Returns the taper function along x and the taper
    function along y
    """
    tapx = np.zeros(ntx)
    if(ntx > 0):
      tapx = np.asarray([np.sin(0.5*np.pi*it/ntx) for it in range(ntx)],dtype='float32')
      tapx = (tapx+1)/2

    tapy = np.zeros(nty)
    if(nty > 0):
      tapy = np.asarray([np.sin(0.5*np.pi*it/nty) for it in range(nty)],dtype='float32')
      tapy = (tapy+1)/2

    return tapx,tapy

  def build_karray(self,dx,dy,bx,by) -> numpy.ndarray:
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
    # Spatial frequency axes
    dkx = 2*np.pi/(bx*dx)
    okx = 0 if bx == 1 else -np.pi/dx
    dky = 2*np.pi/(by*dy)
    oky = 0 if by == 1 else -np.pi/dy

    kk = np.zeros([by,bx],dtype='float32')

    # Populate the array
    for iy in range(by):
      jy = iy+by/2 if iy < by/2 else iy-by/2
      for ix in range(bx):
        jx = ix+bx/2 if ix < bx/2 else ix-bx/2
        kx = okx + jx + dkx
        kk[iy,ix] = kx*kx + ky*ky

    return kk
  
  def nrefs(self, nrmax, dsmax, slo, sloref):
    """
    Computes the number of reference velocities at each depth

    Parameters:
      nrmax  - maximum number of reference velocities
      dsmax  - maximum change in slowness with depth?
      ns     - number of slowness per depth
      slo    - input slowness array
      sloref - number of reference slownesses with depth

    Returns the number of reference slowness with depth.
    Also updates slowref
    """
    ns      = np.prod(slo.shape)
    slo2    = np.zeros(ns,dtype='float32')
    slo2[:] = slo.flatten()[:]
    smax = self.quantile(ns-1,slo2)
    smin = self.quantile(0   ,slo2)
    nrmax = nrmax if nrmax < 1 + (smax-smin)/ds else 1+(smax-smin)/ds

    jr,s2 = 0,0.0
    for ir in range(nrmax):
      qr = (ir + 1.0)/nrmax - 0.5 * 1/nrmax
      s = quantile(qr*ns,slo2)
      if(ir == 0 or np.abs(s-s2) > ds):
        sloref[jr] = s*s
        s2 = s
        jr += 1

    return jr

  def quantile(self, q, a):
    """

    Parameters:
      q - position of interest in the array
      a - input slowness array

    Returns
    """
    n = a.shape[0]
    lo,hi,k = 0,n-1,q
    while(lo < hi):
      ak = a[k]
      i,j = lo,hi
      while(True):
        while(a[i] < ak): i += 1
        while(a[j] > ak): j -= 1
        if(i <= j):
          buf  = a[i]
          a[i] = a[j]
          a[j] = buf
          i += 1; j -= 1
        if(i > j): break
      if(j < k): low = i
      if(k < i): hi  = j
    return a[k]

