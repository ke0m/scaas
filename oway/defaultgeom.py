"""
Default geometry for synthetics
Sources and receivers on the surface
and distributed evenly across the surface
@author: Joseph Jennings
@version: 2020.06.25
"""
import numpy as np
from oway.ssr3 import ssr3
from utils.ptyprint import printprogress
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
    # Source gometry
    self.__nsx = nsx; self.__osx = osx; self.__dsx = dsx
    self.__nsy = nsy; self.__osy = osy; self.__dsy = dsy
    # Frequency axis
    self.__nwo = None; self.__ow = None; self.__dw = None;

  def model_data(self,wav,dt,t0,minf,maxf,vel,ref,nrmax=3,eps=0.01,dtmax=5e-05,time=True,
                 ntx=0,nty=0,px=0,py=0,verb=True):
    """
    Models single scattered (Born) data with the one-way
    wave equation (single square root (SSR), split-step Fourier method).

    Parameters:
      wav   - the input wavelet (source time function) [nt]
      dt    - sampling interval of wavelet
      t0    - time-zero of wavelet (e.g., peak of ricker wavelet)
      minf  - minimum frequency to propagate [Hz]
      maxf  - maximum frequency to propagate [Hz]
      vel   - input velocity model [nz,ny,nx]
      ref   - input reflectivity model [nz,ny,nx]
      nrmax - maximum number of reference velocities [3]
      eps   - stability parameter [0.01]
      dtmax - maximum time error [5e-05]
      time  - return the data back in the time domain [True]
      ntx   - size of taper in x direction (samples) [0]
      nty   - size of taper in y direction (samples) [0]
      px    - amount of padding in x direction (samples)
      py    - amount of padding in y direction (samples)
      verb  - verbosity flag
    
    Returns the data at the surface (in time or frequency) [nw,nry,nrx]
    """
    # Save wavelet temporal parameters
    nt = wav.shape[0]; it0 = int(t0/dt)

    # Create the input frequency domain source and get original frequency axis
    self.__nwo,self.__ow,self.__dw,wfft = self.convert_source(dt,wav,minf=minf,maxf=maxf)
    self.__nwc = wfft.shape[0] # Get the number of frequencies to compute

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz,     # Spatial Samplings
               self.__nwc,self.__ow,self.__dw,eps, # Frequency axis
               ntx,nty,px,py,                      # Taper and padding
               dtmax,nrmax)                        # Reference velocities

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    # Allocate output data (surface wavefield)
    datw = np.zeros([self.__nsy,self.__nsx,self.__nwc,self.__ny,self.__nx],dtype='complex64')

    # Allocate the source for one shot
    sou = np.zeros([self.__nwc,self.__ny,self.__nx],dtype='complex64')

    # Loop over sources
    if(verb):
      k = 0; tot = self.__nsx*self.__nsy;
    for isy in range(self.__nsy):
      sy = int(self.__osy + isy*self.__dsy)
      for isx in range(self.__nsx):
        if(verb): printprogress("nexp:",k,tot)
        sx = int(self.__osx + isx*self.__dsx)
        # Create the source for this shot
        sou[:] = 0.0; 
        sou[:,sy,sx]  = wfft[:]
        # Downward continuation
        ssf.modallw(ref,sou,datw[isy,isx,:,:,:])
        if(verb): k += 1
    if(verb): printprogress("nexp:",tot,tot)

    if(time):
      # Inverse fourier transform
      datt = self.convert_data(datw,self.__nwo,self.__ow,self.__dw,nt,it0)
      return datt
    else:
      return datw

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

  #TODO: handle the source and the data
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
    nw = int(nt/2+1)
    dw = 1/(nt*dt)
    # Min and max frequencies
    begw = int(minf/dw); endw = int(maxf/dw)
    wavp = np.pad(wav,(0,nt),mode='constant')
    wfft = np.fft.fft(wav)[begw:endw]

    return nw,minf,dw,wfft

  def convert_data(self,dat,nw,ow,dw,nt,it0=None):
    """
    Converts the data from frequency to time

    Parameters:
      dat - input data [nw/t,ny,nx]
      nw  - original number of frequencies
      ow  - frequency origin (minf)
      dw  - frequency sampling interval
      nt  - output number of time samples
      it0 - sample index of t0 [0]
    """
    # Get number of computed frequencies
    nwc = dat.shape[2]
    # Transpose the data so frequency is on fast axis
    datt = np.transpose(dat,(0,1,3,4,2)) # [nsy,nsx,nwc,ny,nx] -> [nsy,nsx,ny,nx,nwc]
    # Pad to the original frequency range
    padb = int(ow/dw); pade = nw - nwc - padb
    dattpad = np.pad(datt,((0,0),(0,0),(0,0),(0,0),(padb,pade)),mode='constant') # [*,nwc] -> [*,nw]
    # Inverse FFT and window to t0 (wavelet shift)
    datf2t = np.real(np.fft.ifft(dattpad))
    if(it0 is not None):
      datf2tw = datf2t[:,:,:,:,it0:]
    else:
      datf2tw = datf2t
    # Pad and transpose
    datf2tp = np.pad(datf2tw,((0,0),(0,0),(0,0),(0,0),(0,nt-(nw-it0))),mode='constant')
    odat = np.transpose(datf2tp,(0,1,4,2,3)) # [nsy,nsx,ny,nx,nt] -> [nsy,nsx,nt,ny,nx]
     
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

