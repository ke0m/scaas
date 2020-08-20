"""
Utility functions for modeling and imaging with
one-way wave equation

@author: Joseph Jennings
@version: 2020.08.18
"""
import numpy as np
from oway.ssr3 import interp_slow

def interp_vel(nz,ny,oy,dy,nx,ox,dx,
               velin,dvx,dvy,ovx=0.0,ovy=0.0):
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
  [nvz,nvy,nvx] = velin.shape
  if(nvz != nz):
    raise Exception("Slowness depth axis must be same as output image")
  else:
    nvz = nz

  # Output slowness
  velot = np.zeros([nz,ny,nx],dtype='float32')

  interp_slow(nz,           # Depth samples
              nvy,ovy,dvy,  # Slowness y axis
              nvx,ovx,dvx,  # Slowness x axis
              ny,oy,dy,     # Image y axis
              nx,ox,dx,     # Image x axis
              velin,velot)  # Inputs and outputs

  return velot

def make_sht_cube(nrec,dat):
  """
  Makes a regular cube of shots from the input traces.
  Assumes that the data are already sorted by common
  shot

  Note only works for 2D data at the moment

  Parameters:
    nrec - the number of receivers per shot
    dat  - input shot data [ntr,nt]

  Returns:
    regular shot cube [nsht,nrx,nt]
  """
  # Get data dimensions
  if(dat.ndim != 2):
    raise Exception("Data must be of dimension [ntr,nt]")
  nt = dat.shape[1]

  # Get maximum number of receivers
  nrecxmax = np.max(nrec)

  # Get number of shots
  nexp = len(nrec)

  # Output shot array
  shots = np.zeros([nexp,nrecxmax,nt],dtype='float32')

  # Loop over all sources
  ntr = 0
  for iexp in range(nexp):
    shots[iexp,:nrec[iexp],:] = dat[ntr:ntr+nrec[iexp],:]
    ntr += nrec[iexp]

  return shots

def fft1(sig,dt,minf,maxf):
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
  nt = 2*next_fast_size(int((n1+1)/2))
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

def ifft1(sig,nw,ow,dw,n1,it0=0):
  """
  Computes the IFFT along the fast axis. Input
  array can be N-dimensional

  Parameters:
    sig - input frequency-domain signal (frequency is fast axis)
    nw  - original number of frequencies
    ow  - frequency origin (minf)
    dw  - frequency sampling interval
    n1  - output number of time samples
    it0 - sample index of t0 [0]
  """
  # Get number of computed frequencies
  nwc = sig.shape[-1]
  # Compute size for FFT
  nt = 2*(nw-1)
  # Pad to the original frequency range
  padb = int(ow/dw); pade = nw - nwc - padb
  paddims1 = [(0,0)]*(sig.ndim-1)
  paddims1.append((padb,pade))
  sigp1   = np.pad(sig,paddims1,mode='constant')
  # Pad for the inverse FFT
  paddims2 = [(0,0)]*(sigp1.ndim-1)
  paddims2.append((0,nt-nw))
  sigp2     = np.pad(sigp1,paddims2,mode='constant')
  sigifft   = np.real(np.fft.ifft(sigp2))
  sigifftw  = sigifft[...,it0:]
  # Pad to desired output time samples
  paddims3 = [(0,0)]*(sigifftw.ndim-1)
  paddims3.append((0,n1-(nt-it0)))
  sigifftwp = np.pad(sigifftw,paddims3,mode='constant')

  return sigifftwp

def next_fast_size(n):
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

