"""
Default geometry for synthetics
Sources and receivers on the surface
and distributed evenly across the surface
@author: Joseph Jennings
@version: 2020.06.25
"""
import numpy as np
from oway.ssr3 import ssr3
import matplotlib.pyplot as plt

class defaultgeom:
  """
  Functions for modeling and imaging with a
  standard synthetic source receiver geometry
  """
  def __init__(self,nx,dx,ny,dy,nz,dz,                             # Model size
               nsx,dsx,nsy,dsy,osx=0.0,osy=0.0,                    # Source geometry
               nrx=None,drx=1.0,orx=0.0,nry=None,dry=1.0,ory=0.0): # Receiver geometry
    """
    Creates a default geometry object for split-step fourier downward continuation

    Parameters:
      nx    - Number of x samples of the velocity model
      dx    - x sampling of the velocity model
      ny    - Number of y samples of the velocity model
      dy    - y sampling of the velocity model
      nz    - Number of z samples of the velocity model
      dz    - z sampling of the velocity model
      nsx   - Total number of sources in x direction
      dsx   - Spacing between sources along x direction (in samples)
      osx   - x-sample coordinate of first source [0.0]
      nsy   - Total number of sources in y direction
      dsy   - Spacing between sources along y diection (in samples)
      osy   - y-sample coordinate of first source [0.0]
      nrx   - Total number of receivers in x direction [One for every surface location]
      drx   - Spacing between receivers along x direction (in samples) [1.0]
      orx   - x-sample coordinate of first receiver [0.0]
      nry   - Total number of receivers in y direction [One for every surface location]
      dry   - Spacing between receivers along y direction (in samples) [1.0]
      ory   - y-sample coordinate of first receiver [0.0]

    Returns:
      a default geom object
    """
    # Spatial axes
    self.__nx = nx; self.__dx = dx
    self.__ny = ny; self.__dy = dy
    self.__nz = nz; self.__dz = dz
    # Frequency axis
    self.__nw   = None; self.__dw   = None;
    self.__begw = None; self.__endw = None;

  def model_data(self,wav,minf,maxf,vel,ref):
    """
    """
    pass

  def inject_source(self,ix,iy,wav):
    """
    Injects the frequency domain source at the ix
    and iy sample position within the model

    Parameters:
      ix  - x sample position for source injection
      iy  - y sample position for source injection
      wav - frequency domain source to be injected

    Returns the frequency domain source at the surface [nw,ny,nx]
    """
    nw = wav.shape[0]
    sou = np.zeros([nw,self.__ny,self.__nx],dtype='complex64')
    sou[:,iy,ix] = wav[:]

    return sou

  def convert_source(self,dt,wav,minf,maxf):
    """
    Creates a frequency domain source from a minimum
    to a maximum specified frequency

    Parameters:
      dt   - temporal sampling of wavelet
      wav  - the input source time function [nt]
      minf - the lowest frequency to propagate  [Hz]
      maxf - the highest frequency to propagate [Hz]

    Returns:
      a frequency domain source and the frequency axis [nw]
    """
    # Get sizes for time and frequency domain
    n1 = wav.shape[0]
    nt = 2*self.next_fast_size(int((n1+1)/2))
    if(nt%2): nt += 1
    self.__nw = int(nt/2+1)
    self.__dw = 1/(nt*dt)
    # Min and max frequencies
    self.__begw = int(minf/self.__dw) 
    self.__endw = int(maxf/self.__dw)
    wavp = np.pad(wav,(0,nt),mode='constant')
    wfft = np.fft.fft(wav)[self.__begw:self.__endw]

    return wfft,minf,self.__dw

  def convert_data(self,d1,dat,f2t=True):
    """
    Converts the data from either time to frequency or back

    Parameters:
      d1  - time or frequency sampling of data
      dat - input data [nw/t,ny,nx]
      f2t - flag indicating frequency to time [True]
    """
    if(f2t):
      # Get number of frequencies
      n1 = dat.shape[0]
      # Transpose the data so frequency is on fast axis
      datt = np.transpose(dat,(1,2,0))
      if(self.__nw is not None and self.__begw is not None):
        # Pad to the original frequency range
        padb = self.__begw; pade = self.__nw - n1 - self.__begw
        dattpad = np.pad(datt,((0,0),(0,0),(padb,pade)),mode='constant')
        # Inverse FFT
        odat = np.fft.ifft1(dattpad)
     
    return odat

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

