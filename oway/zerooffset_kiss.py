"""
Class for imaging zero-offset data 
with the one-way wave equation
@author: Joseph Jennings
@version: 2020.07.16
"""
import numpy as np
from oway.ssr3_kiss import ssr3, interp_slow
from utils.ptyprint import progressbar
import matplotlib.pyplot as plt

class zerooffset:
  """
  Functions for modeling and imaging with a
  standard synthetic source receiver geometry
  """
  def __init__(self,nx,dx,ny,dy,nz,dz,ox=0.0,oy=0.0):
    """
    Creates a default geometry object for split-step fourier downward continuation

    Parameters:
      nx    - Number of x samples of the velocity model
      dx    - x sampling of the velocity model
      ny    - Number of y samples of the velocity model
      dy    - y sampling of the velocity model
      nz    - Number of z samples of the velocity model
      dz    - z sampling of the velocity model
      ox    - Origin of x axis
      oy    - Origin of y axis

    Returns:
      a zero offset object
    """
    # Spatial axes
    self.__nx = nx; self.__dx = dx; self.__ox = ox
    self.__ny = ny; self.__dy = dy; self.__oy = oy
    self.__nz = nz; self.__dz = dz

    # Frequency axis
    self.__nwo = None; self.__ow = None; self.__dw = None;

  def get_freq_axis(self):
    """ Returns the frequency axis """
    return self.__nwc,self.__ow,self.__dw

  def interp_vel(self,velin,dvx,dvy,ovx=0.0,ovy=0.0):
    """
    Lateral nearest-neighbor interpolation of velocity. Use
    this when imaging grid is different than velocity
    grid. Assumes the same depth axis for imaging
    and slowness grid

    Parameters:
      velin - the input velocity field [nz,nvy,nvx]
      dvy   - the y sampling of the slowness field
      dvx   - the x sampling of the slowness field
      ovy   - the y origin of the slowness field [0.0]
      ovx   - the x origin of the slowness field [0.0]

    Returns:
      the interpolated velocity field now same size
      as output imaging grid [nz,ny,nx]
    """
    # Get dimensions
    [nz,nvy,nvx] = velin.shape
    if(nz != self.__nz):
      raise Exception("Slowness depth axis must be same as output image")

    # Output slowness
    velot = np.zeros([nz,self.__ny,self.__nx],dtype='float32')

    interp_slow(self.__nz,                     # Depth saples
                nvy,ovy,dvy,                   # Slowness y axis
                nvx,ovx,dvx,                   # Slowness x axis
                self.__ny,self.__oy,self.__dy, # Image y axis
                self.__nx,self.__ox,self.__dx, # Image x axis
                velin,velot)                   # Inputs and outputs

    return velot

  def model_data(self,img,dt,minf,maxf,vel,ref,jf=1,nrmax=3,eps=0.01,dtmax=5e-05,time=True,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,wverb=True):
    """
    3D modeling of zero-offset data with the one-way
    wave equation (single square root (SSR), split-step Fourier method).

    Parameters:
      wav    - the input migrated zero-offset image [nz,ny,nx]
      dt     - sampling interval of output data
      minf   - minimum frequency to propagate [Hz]
      maxf   - maximum frequency to propagate [Hz]
      vel    - input velocity model [nz,ny,nx]
      ref    - input reflectivity model [nz,ny,nx]
      jf     - frequency decimation factor [1]
      nrmax  - maximum number of reference velocities [3]
      eps    - stability parameter [0.01]
      dtmax  - maximum time error [5e-05]
      time   - return the data back in the time domain [True]
      ntx    - size of taper in x direction (samples) [0]
      nty    - size of taper in y direction (samples) [0]
      px     - amount of padding in x direction (samples)
      py     - amount of padding in y direction (samples)
      nthrds - number of OpenMP threads for parallelizing over frequency [1]
      wverb  - verbosity flag for frequency progressbar [True]
    
    Returns the zero-offset data (stack) (in time or frequency) [nw/nt,ny,nx]
    """
    pass

  def image_data(self,dat,dt,minf,maxf,vel,jf=1,nrmax=3,eps=0.01,dtmax=5e-05,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,wverb=True):
    """
    3D migration of zero-offset data via the one-way wave equation (single-square
    root split-step fourier method)

    Parameters:
      dat    - input shot profile data [ny,nx,nt]
      dt     - temporal sampling of input data
      minf   - minimum frequency to image in the data [Hz]
      maxf   - maximum frequency to image in the data [Hz]
      vel    - input migration velocity model [nz,ny,nx]
      jf     - frequency decimation factor
      nrmax  - maximum number of reference velocities [3]
      eps    - stability parameter [0.01]
      dtmax  - maximum time error [5e-05]
      ntx    - size of taper in x direction [0]
      nty    - size of taper in y direction [0]
      px     - amount of padding in x direction (samples) [0]
      py     - amount of padding in y direction (samples) [0]
      nthrds - number of OpenMP threads for parallelizing over frequency [1]
      wverb  - verbosity flag for frequency progress bar [True]

    Returns:
      an image created from the data [nz,ny,nx]
    """
    # Get temporal axis
    nt = dat.shape[-1]

    # Create frequency domain source
    wav    = np.zeros(nt,dtype='float32')
    wav[0] = 1.0
    self.__nwo,self.__ow,self.__dw,wfft = self.fft1(wav,dt,minf=minf,maxf=maxf)
    wfftd = wfft[::jf]
    self.__nwc = wfftd.shape[0] # Get the number of frequencies for imaging
    self.__dwc = self.__dw*jf

    if(wverb): print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))

    # Create frequency domain data
    _,_,_,dfft = self.fft1(dat,dt,minf=minf,maxf=maxf)
    dfftd = dfft[:,::jf]
    datt = np.transpose(dfftd,(2,0,1)) # [ny,nx,nwc] -> [nwc,ny,nx] 
    datw = np.ascontiguousarray(datt.reshape([self.__nwc,self.__ny,self.__nx]))

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz ,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz ,     # Spatial Samplings
               self.__nwc,self.__ow,self.__dwc,eps, # Frequency axis
               ntx,nty,px,py,                       # Taper and padding
               dtmax,nrmax)                         # Reference velocities

    # Output image
    img = np.zeros([self.__nz,self.__ny,self.__nx],dtype='float32')

    # Compute slowness and reference slownesses
    slo = 2/vel # Two-way travel time
    ssf.set_slows(slo)

    # Image for all frequencies
    ssf.migallwzo(datw,img,nthrds,wverb)

    return img

  def fft1(self,sig,dt,minf,maxf):
    """
    Computes the FFT along the fast axis. Input
    array can be N-dimensional

    Parameters:
      sig  - the input time-domain signal (time is fast axis)
      dt   - temporal sampling of input data
      minf - the minimum frequency for windowing the spectrum [Hz]
      maxf - the maximum frequency for windowing the spectrum

    Returns: 
      the frequency domain data (frequency is fast axis) and the
      frequency axis [nw,ow,dw]
    """
    n1 = sig.shape[-1]
    nt = 2*self.next_fast_size(int((n1+1)/2))
    if(nt%2): nt += 1
    nw = int(nt/2+1)
    dw = 1/(nt*dt)
    # Min and max frequencies
    begw = int(minf/dw); endw = int(maxf/dw)
    # Create the padded dimensions (only last axis)
    paddims = [(0,0)]*(sig.ndim-1)
    paddims.append((0,nt-n1))
    sigp   = np.pad(sig,paddims,mode='constant')
    # Compute the FFT
    sigfft = np.fft.fft(sigp)[...,begw:endw]

    return nw,minf,dw,sigfft.astype('complex64')

  def data_f2t(self,dat,nw,ow,dw,n1,it0=None):
    """
    Converts the data from frequency to time

    Parameters:
      dat - input data [nw,ny,nx]
      nw  - original number of frequencies
      ow  - frequency origin (minf)
      dw  - frequency sampling interval
      n1  - output number of time samples
      it0 - sample index of t0 [0]
    """
    # Get number of computed frequencies
    nwc = dat.shape[2]
    # Compute size for FFT
    nt = 2*(nw-1)
    # Transpose the data so frequency is on fast axis
    datt = np.transpose(dat,(0,1,3,4,2)) # [nsy,nsx,nwc,ny,nx] -> [nsy,nsx,ny,nx,nwc]
    # Pad to the original frequency range
    padb = int(ow/dw); pade = nw - nwc - padb
    dattpad  = np.pad(datt,((0,0),(0,0),(0,0),(0,0),(padb,pade)),mode='constant')  # [*,nwc] -> [*,nw]
    # Pad for the inverse FFT
    dattpadp = np.pad(dattpad,((0,0),(0,0),(0,0),(0,0),(0,nt-nw)),mode='constant') # [*,nw] -> [*,nt]
    # Inverse FFT and window to t0 (wavelet shift)
    datf2t = np.real(np.fft.ifft(dattpadp))
    if(it0 is not None):
      datf2tw = datf2t[:,:,:,:,it0:]
    else:
      datf2tw = datf2t
    # Pad and transpose
    datf2tp = np.pad(datf2tw,((0,0),(0,0),(0,0),(0,0),(0,n1-(nt-it0))),mode='constant')

    return datf2tp

  def next_fast_size(self,n):
    """ Gets the optimal size for computing the FFT """
    while(1):
      m = n
      while( (m%2) == 0 ): m/=2
      while( (m%3) == 0 ): m/=3
      while( (m%5) == 0 ): m/=5
      if(m<=1):
        break
      n += 1

    return n

