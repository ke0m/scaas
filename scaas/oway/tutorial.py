"""
Functions for a one-way wave equation tutorial

@author: Joseph Jennings
@version: 2021.01.07
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
    self.__nz,self.__ny,self.__nx = nz,ny,nx
    self.__dz,self.__dy,self.__dx = dz,dy,dx
    self.__bx = nx + px
    self.__by = ny + py
    # Build the taper
    self.__tap = self.build_taper(ntx,nty)
    # Build spatial frequencies
    self.__kk = self.build_karray(self.__dx,self.__dy,self.__bx,self.__by)

    # Slowness array
    self.__slo = None

    # Reference velocities
    self.__nr     = np.zeros(nz,dtype='int32')
    self.__nrmax  = nrmax
    self.__dsmax  = dtmax/dz
    self.__dsmax2 = self.__dsmax*self.__dsmax*self.__dsmax*self.__dsmax
    self.__sloref = np.zeros([nz,nrmax],dtype='float32')

    # Frquency axes
    self.__nw, self.__ow, self.__dw = nw, 2*np.pi*ow, 2*np.pi*dw
    
    # Wavefield slices
    self.__sslc = None
    self.__rslc = None
    self.__wt   = np.zeros([self.__ny,self.__nx],dtype='float32')
    self.__wxot = np.zeros([self.__ny,self.__nx],dtype='complex64')
    self.__wxx  = np.zeros([self.__ny,self.__nx],dtype='complex64')
    self.__wxk  = np.zeros([self.__by,self.__bx],dtype='complex64')
    self.__wkk  = np.zeros([self.__by,self.__bx],dtype='complex64')

    # Image
    self.__wimg = None

    # Stability parameter
    self.__eps = eps

  def set_slows(self, slo) -> None:
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
    for iz in range(self.__nz):
      self.__nr[iz] = self.nrefs(self.__nrmax, self.__dsmax,slo[iz], self.__sloref[iz])

    # Build reference slownesses
    for iz in range(self.__nz-1):
      self.__sloref[iz] = 0.5*(self.__sloref[iz] + self.__sloref[iz+1])

  def mod_allw(self, ref, wav, verb=False) -> np.ndarray:
    """
    Linearized modeling of all frequencies using a SSR SSF operator 

    Parameters:
      ref  - the reflectivity model [nz,ny,nx]
      wav  - the input wavelet (FT'ed in time) [nw,ny,nx]
      verb - verbosity flag [False]

    Returns:
      Data modeled for all frequencies
    """
    if(self.__slo is None): 
      raise Exception("Must run set_slows before modeling or migration")

    # Output data
    dat = np.zeros([self.__nw,self.__ny,self.__nx],dtype='complex64')

    # Allocate wavefield slices for mod_onew
    self.__sslc = np.zeros([self.__nz,self.__ny,self.__nx],dtype='complex64')
    self.__rslc = np.zeros([self.__ny,self.__nx],dtype='complex64')

    for iw in progressbar(range(self.__nw),"nw:",verb=verb):
      dat[iw] = self.mod_onew(iw, ref, wav[iw])

    return dat

  def mod_onew(self, iw, ref, wav) -> np.ndarray:
    """
    Linearized modeling of one frequency using a SSR SSF operator

    Parameters
      iw  - the frequency index
      ref - reflectivity model [nz,ny,nx]
      wav - frequency slice of the wavelet

    Returns:
      Data extrapolated in depth for one frequency
    """
    w = np.complex(self.__eps*self.__dw,self.__ow + iw*self.__dw)

    # Initialize the wavefield slices
    self.__sslc[:] = 0.0
    self.__rslc[:] = 0.0

    # Boundary condition at z=0
    self.__sslc[0] = self.__tap*wav

    # Source loop over depth
    for iz in range(self.__nz-1):
      # Depth extrapolation
      self.__sslc[iz+1] = self.ssf(w,iz,self.__slo[iz],self.__slo[iz+1],self.__sslc[iz])

    # Receiver loop over depth
    for iz in range(self.__nz-1,0,-1):
      # Scattering with reflectivity
      self.__sslc[iz] *= ref[iz]
      self.__rslc += self.__sslc[iz]

      # Depth extrapolation
      self.__rslc = self.ssf(w,iz,self.__slo[iz],self.__slo[iz-1],self.__rslc)

    return self.__tap*self.__rslc

  def mig_allw(self, dat, wav, verb=True) -> np.ndarray:
    """
    Migration of data using a SSR SSF operator

    Properties:
      dat  - input data [nw,ny,nx]
      wav  - input injected wavelet [nw,ny,nx]
      verb - verbosity flag [True]

    Returns:
      the migrated image [nz,ny,nx]
    """
    if(self.__slo is None):
      raise Exception("Must run set_slows before modeling or migration")

    # Output image
    img         = np.zeros([self.__nz,self.__ny,self.__nx],dtype='float32')
    self.__wimg = np.zeros([self.__nz,self.__ny,self.__nx],dtype='float32')

    # Allocate wavefield slices for mod_onew
    self.__sslc = np.zeros([self.__ny,self.__nx],dtype='complex64')
    self.__rslc = np.zeros([self.__ny,self.__nx],dtype='complex64')

    for iw in progressbar(range(self.__nw),"nw:",verb=verb):
      img += self.mig_onew(iw, dat[iw], wav[iw])

    return img

  def mig_onew(self, iw, dat, wav) -> np.ndarray:
    """
    SSR SSF migration for a single frequency

    Parameters:
      iw  - the frequency index
      dat - the input data for a single frequency [ny,nx]
      wav - the inject wavelet for a single frequency [ny,nx]

    Returns:
      the image migrated for a single frequency [nz,ny,nx]
    """
    ws = np.complex(self.__eps*self.__dw, +(self.__ow + iw*self.__dw)) # Causal
    wr = np.complex(self.__eps*self.__dw, -(self.__ow + iw*self.__dw)) # Anti-causal

    self.__sslc[:,:] = self.__tap*wav
    self.__rslc[:,:] = self.__tap*dat

    for iz in range(self.__nz-1):
      # Depth extrapolation of source and receiver wavefields
      self.__sslc = self.ssf(ws, iz, self.__slo[iz], self.__slo[iz+1], self.__sslc)
      self.__rslc = self.ssf(wr, iz, self.__slo[iz], self.__slo[iz+1], self.__rslc)
      # Imaging condition
      self.__wimg[iz] = np.real(np.conj(self.__sslc) * self.__rslc)

    return self.__wimg

  def ssf(self, w, iz, scur, snex, wxin) -> np.ndarray:
    """
    Extended Split-step Fourier operator for depth extrapolation

    Parameters:
      w    - the frequency (complex)
      iz   - depth index
      scur - slowness slice at current depth (iz, [ny,nx])
      snex - slowness slice at next depth (iz+1, [ny,nx])
      wxin - wavefield (w-x-y) at current depth

    Returns:
      the wavefield extrapolated at the next depth
    """
    # Initialize array
    self.__wt[:] = 0.0

    self.__wxot  = wxin * np.exp(-w*scur*0.5*self.__dz)
    self.__wxot *= 1/np.sqrt(self.__bx*self.__by)

    self.__wxk[:] = 0.0
    self.__wxk[:self.__ny,:self.__nx] = self.__wxot[:,:]
    # x-y -> kx-ky
    self.__wkk = np.fft.fft2(self.__wxk)
    self.__wxot[:] = 0.0

    for ir in range(self.__nr[iz]):
      co = np.complex(0,np.sqrt(np.imag(w)*np.imag(w)*self.__sloref[iz,ir]))*np.sign(np.imag(w))
      cc = self.mysqrt(w*w*self.__sloref[iz,ir] + self.__kk, np.sign(np.imag(w)))
      self.__wxk = self.__wkk * np.exp((co-cc)*self.__dz)

      # kx-ky -> x-y (scale to match with FFTW)
      self.__wxx = np.fft.ifft2(self.__wxk)[:self.__ny,:self.__nx]*(self.__bx*self.__by)

      # Interpolate/accumulate
      d = np.abs(scur*scur - self.__sloref[iz,ir])
      d = self.__dsmax2/(d*d + self.__dsmax2)
      self.__wxot += self.__wxx * d/np.sqrt(self.__by*self.__bx)
      self.__wt   += d

    self.__wxot /= self.__wt
    self.__wxot *= np.exp(-w*snex*0.5*self.__dz)

    return self.__tap*self.__wxot

  def mysqrt(self,x,sign):
    out = np.zeros(x.shape,dtype='complex64')
    idx = x >= 0
    out[idx]  = np.sqrt(x[idx])
    out[~idx] = np.sqrt(-x[~idx])*sign*1j
    return out

  def build_taper(self,ntx,nty) -> np.ndarray:
    """
    Builds a 2D tapering function
  
    Parameters:
      ntx - the size of the taper in the x direction
      nty - the size of the taper in the y direction
  
    Returns the taper function along x and the taper
    function along y
    """
    # Output taper
    tapout = np.ones([self.__ny,self.__nx],dtype='float32')

    # Taper along x
    tapx = np.zeros(ntx)
    if(ntx > 0):
      tapx = np.asarray([np.sin(0.5*np.pi*it/ntx) for it in range(ntx)],dtype='float32')
      tapx = (tapx+1)/2

    for it in range(ntx):
      gain = tapx[it]
      for iy in range(self.__ny):
        tapout[iy,it] *= gain
        tapout[iy,self.__nx-it-1] *= gain

    # Taper along y
    tapy = np.zeros(nty)
    if(nty > 0):
      tapy = np.asarray([np.sin(0.5*np.pi*it/nty) for it in range(nty)],dtype='float32')
      tapy = (tapy+1)/2

    for it in range(nty):
      gain = tapy[it]
      for ix in range(self.__nx):
        tapout[it,ix] *= gain
        tapout[self.__ny-it-1,ix] *= gain

    return tapout

  def build_karray(self,dx,dy,bx,by) -> np.ndarray:
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
      jy = iy+by//2 if iy < by//2 else iy-by//2
      ky = oky + jy*dky
      for ix in range(bx):
        jx = ix+bx//2 if ix < bx//2 else ix-bx//2
        kx = okx + jx*dkx
        kk[iy,ix] = kx*kx + ky*ky

    return kk
  
  def nrefs(self, nrmax, ds, slo, sloref) -> int:
    """
    Computes the number of reference velocities at each depth

    Parameters:
      nrmax  - maximum number of reference velocities
      ds     - maximum change in slowness with depth?
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
    nrmax = nrmax if nrmax < int(1+(smax-smin)/ds) else int(1+(smax-smin)/ds)

    jr,s2 = 0,0.0
    for ir in range(nrmax):
      qr = (ir + 1.0)/nrmax - 0.5 * 1/nrmax
      s = self.quantile(int(qr*ns),slo2)
      if(ir == 0 or np.abs(s-s2) > ds):
        sloref[jr] = s*s
        s2 = s
        jr += 1

    return jr

  def quantile(self, q, a) -> float:
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
      if(j < k): lo  = i
      if(k < i): hi  = j
    return a[k]

