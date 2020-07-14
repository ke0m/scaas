"""
Imaging/modeling data based on source
and receiver coordinates
@author: Joseph Jennings
@version: 2020.07.07
"""
import numpy as np
from oway.ssr3 import ssr3
from utils.ptyprint import progressbar
import matplotlib.pyplot as plt

class coordgeom:
  """
  Functions for modeling and imaging with a
  field data (coordinate) geometry
  """
  def __init__(self,nx,dx,ny,dy,nz,dz,nrec,srcxs=None,srcys=None,recxs=None,recys=None,
               ox=0.0,oy=0.0,oz=0.0):
    """
    Creates a coordinate geometry object for split-step fourier downward continuation.
    Expects that the coordinates are integer sample number coordinates (already divided by dx or dy)

    Parameters:
      nx    - Number of x samples of the velocity model
      dx    - x sampling of the velocity model
      ny    - Number of y samples of the velocity model
      dy    - y sampling of the velocity model
      nz    - Number of z samples of the velocity model
      dz    - z sampling of the velocity model
      nrec  - number of receivers per shot (int) [number of shots]
      srcxs - x coordinates of source locations [number of shots]
      srcys - y coordinates of source locations [number of shots]
      recxs - x coordinates of receiver locations [number of traces]
      recys - y coordinates of receiver locations [number of traces]

    Returns:
      a coordinate geom object
    """
    # Spatial axes
    self.__nx = nx; self.__ox = ox; self.__dx = dx
    self.__ny = ny; self.__oy = oy; self.__dy = dy
    self.__nz = nz; self.__oz = oz; self.__dz = dz
    ## Source gometry
    # Check if either is none
    if(srcxs is None and srcys is None):
      raise Exception("Must provide either srcx or srcy coordinates")
    if(srcxs is None):
      srcxs = np.zeros(len(srcys),dtype='int')
    if(srcys is None):
      srcys = np.zeros(len(srcxs),dtype='int')
    # Make sure coordinates are within the model
    if(np.any(srcxs >= ox+(nx)*dx) or np.any(srcys >= oy+(ny)*dy)):
      raise Exception("Source geometry must be within model size")
    if(np.any(srcxs < ox) or np.any(srcys <  oy)):
      raise Exception("Source geometry must be within model size")
    if(len(srcxs) != len(srcys)):
      raise Exception("Length of srcxs must equal srcys")
    self.__srcxs = srcxs.astype('float32'); self.__srcys = srcys.astype('float32')
    # Total number of sources
    self.__nexp = len(srcxs) 
    ## Receiver geometry
    # Check if either is none
    if(recxs is None and recys is None):
      raise Exception("Must provide either recx or recy coordinates")
    if(recxs is None):
      recxs = np.zeros(len(recys),dtype='int')
    if(recys is None):
      recys = np.zeros(len(recxs),dtype='int')
    # Make sure coordinates are within the model
    if(np.any(recxs >= ox + nx*dx) or np.any(recys >= oy + ny*dy)):
      raise Exception("Receiver geometry must be within model size")
    if(np.any(recxs < ox) or np.any(recys <  oy)):
      raise Exception("Receiver geometry must be within model size")
    if(len(recxs) != len(recys)):
      raise Exception("Each trace must have same number of x and y coordinates")
    self.__recxs = recxs.astype('float32'); self.__recys = recys.astype('float32')
    # Number of receivers per shot
    if(nrec.dtype != 'int'):
      raise Exception("nrec (number of receivers) must be integer type array")
    self.__nrec = nrec
    # Number of traces
    self.__ntr = len(recxs)

    # Frequency axis
    self.__nwo = None; self.__ow  = None; self.__dw = None;
    self.__nwc = None; self.__dwc = None

  def get_freq_axis(self):
    """ Returns the frequency axis """
    return self.__nwc,self.__ow,self.__dw

  def model_data(self,wav,dt,t0,minf,maxf,vel,ref,jf=1,nrmax=3,eps=0.01,dtmax=5e-05,time=True,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=True):
    """
    3D modeling of single scattered (Born) data with the one-way
    wave equation (single square root (SSR), split-step Fourier method).

    Parameters:
      wav    - the input wavelet (source time function) [nt]
      dt     - sampling interval of wavelet
      t0     - time-zero of wavelet (e.g., peak of ricker wavelet)
      minf   - minimum frequency to propagate [Hz]
      maxf   - maximum frequency to propagate [Hz]
      vel    - input velocity model [nz,ny,nx]
      ref    - input reflectivity model [nz,ny,nx]
      jf     - frequency decimation factor
      nrmax  - maximum number of reference velocities [3]
      eps    - stability parameter [0.01]
      dtmax  - maximum time error [5e-05]
      time   - return the data back in the time domain [True]
      ntx    - size of taper in x direction (samples) [0]
      nty    - size of taper in y direction (samples) [0]
      px     - amount of padding in x direction (samples)
      py     - amount of padding in y direction (samples)
      nthrds - number of OpenMP threads to use for frequency parallelization [1]
      sverb  - verbosity flag for shot progress bar [True]
      wverb  - verbosity flag for frequency progress bar [False]
    
    Returns: 
      the data at the surface (in time or frequency) [nw,nry,nrx]
    """
    # Save wavelet temporal parameters
    nt = wav.shape[0]; it0 = int(t0/dt)

    # Create the input frequency domain source and get original frequency axis
    self.__nwo,self.__ow,self.__dw,wfft = self.fft1(wav,dt,minf=minf,maxf=maxf)
    wfftd = wfft[::jf]
    self.__nwc = wfftd.shape[0] # Get the number of frequencies to compute
    self.__dwc = jf*self.__dw

    if(sverb or wverb): print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz ,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz ,     # Spatial Samplings
               self.__nwc,self.__ow,self.__dwc,eps, # Frequency axis
               ntx,nty,px,py,                       # Taper and padding
               dtmax,nrmax)                         # Reference velocities

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    # Allocate output data (surface wavefield) and receiver data
    datw  = np.zeros([self.__nwc,self.__ny,self.__nx],dtype='complex64')
    recw  = np.zeros([self.__ntr,self.__nwc],dtype='complex64')

    # Allocate the source for one shot
    sou = np.zeros([self.__nwc,self.__ny,self.__nx],dtype='complex64')

    # Loop over sources
    ntr = 0
    for iexp in progressbar(range(self.__nexp),"nexp:"):
      # Get the source coordinates
      sy = self.__srcys[iexp]; sx = self.__srcxs[iexp]
      isy = int((sy-self.__oy)/self.__dy+0.5); isx = int((sx-self.__ox)/self.__dx+0.5)
      # Create the source for this shot
      sou[:] = 0.0; 
      sou[:,isy,isx]  = wfft[:]
      # Downward continuation
      datw[:] = 0.0
      ssf.modallw(ref,sou,datw,nthrds,wverb)
      # Restrict to receiver locations
      datwt = np.ascontiguousarray(np.transpose(datw,(1,2,0)))  # [nwc,ny,nx] -> [ny,nx,nwc]
      ssf.restrict_data(self.__nrec[iexp],self.__recys[ntr:],self.__recxs[ntr:],self.__oy,self.__ox,datwt,recw[ntr:,:])
      # Increase number of traces
      ntr += self.__nrec[iexp]

    if(time):
      # Inverse fourier transform
      #TODO: need to modify this program (how to handle frequency decimation?)
      rect = self.data_f2t(recw,self.__nwo,self.__ow,self.__dw,nt,it0)
      return rect
    else:
      return recw

  def image_data(self,dat,dt,minf,maxf,vel,jf=1,nhx=0,nhy=0,sym=True,nrmax=3,eps=0.01,dtmax=5e-05,wav=None,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=True):
    """
    3D migration of shot profile data via the one-way wave equation (single-square
    root split-step fourier method). Input data are assumed to follow
    the default geometry (sources and receivers on a regular grid)

    Parameters:
      dat    - input shot profile data [ntr,nt]
      dt     - temporal sampling of input data
      minf   - minimum frequency to image in the data [Hz]
      maxf   - maximum frequency to image in the data [Hz]
      vel    - input migration velocity model [nz,ny,nx]
      jf     - frequency decimation factor [1]
      nhx    - number of subsurface offsets in x to compute [0]
      nhy    - number of subsurface offsets in y to compute [0]
      sym    - symmetrize the subsurface offsets [True]
      nrmax  - maximum number of reference velocities [3]
      eps    - stability parameter [0.01]
      dtmax  - maximum time error [5e-05]
      wav    - input wavelet [None,assumes an impulse at zero lag]
      ntx    - size of taper in x direction [0]
      nty    - size of taper in y direction [0]
      px     - amount of padding in x direction (samples) [0]
      py     - amount of padding in y direction (samples) [0]
      nthrds - number of OpenMP threads for frequency parallelization [1]
      sverb  - verbosity flag for shot progress bar [True]
      wverb  - verbosity flag for frequency progress bar [False]

    Returns:
      an image created from the data [nhy,nhx,nz,ny,nx]
    """
    # Make sure data are same size as coordinates
    if(dat.shape[0] != self.__ntr):
      raise Exception("Data must have same number of traces passed to constructor")

    # Get temporal axis
    nt = dat.shape[-1]

    # Create frequency domain source
    if(wav is None):
      wav    = np.zeros(nt,dtype='float32')
      wav[0] = 1.0
    self.__nwo,self.__ow,self.__dw,wfft = self.fft1(wav,dt,minf=minf,maxf=maxf)
    wfftd = wfft[::jf]
    self.__nwc = wfftd.shape[0] # Get the number of frequencies for imaging
    self.__dwc = self.__dw*jf

    if(sverb or wverb): print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))

    # Create frequency domain data
    _,_,_,dfft = self.fft1(dat,dt,minf=minf,maxf=maxf)
    dfftd = dfft[:,::jf]
    # Allocate the data for one shot
    datw = np.zeros([self.__ny,self.__nx,self.__nwc],dtype='complex64')
    # Allocate the source for one shot
    sou = np.zeros([self.__nwc,self.__ny,self.__nx],dtype='complex64')

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz ,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz ,     # Spatial Samplings
               self.__nwc,self.__ow,self.__dwc,eps, # Frequency axis
               ntx,nty,px,py,                       # Taper and padding
               dtmax,nrmax)                         # Reference velocities

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    # Allocate partial image array 
    if(nhx == 0 and nhy == 0):
      imgar = np.zeros([self.__nexp,self.__nz,self.__ny,self.__nx],dtype='float32')
    else:
      if(sym):
        imgar = np.zeros([self.__nexp,2*nhy+1,2*nhx+1,self.__nz,self.__ny,self.__nx],dtype='float32')
      else:
        imgar = np.zeros([self.__nexp,nhy+1,nhx+1,self.__nz,self.__ny,self.__nx],dtype='float32')

    # Loop over sources
    ntr = 0
    for iexp in progressbar(range(self.__nexp),"nexp:"):
      # Get the source coordinates
      sy = self.__srcys[iexp]; sx = self.__srcxs[iexp]
      isy = int((sy-self.__oy)/self.__dy+0.5); isx = int((sx-self.__ox)/self.__dx+0.5)
      # Create the source wavefield for this shot
      sou[:] = 0.0
      sou[:,isy,isx]  = wfftd[:]
      # Inject the data for this shot
      datw[:] = 0.0
      ssf.inject_data(self.__nrec[iexp],self.__recys[ntr:],self.__recxs[ntr:],self.__oy,self.__ox,dfftd[ntr:,:],datw)
      datwt = np.ascontiguousarray(np.transpose(datw,(2,0,1))) # [ny,nx,nwc] -> [nwc,ny,nx]
      if(nhx == 0 and nhy == 0):
        # Conventional imaging
        ssf.migallw(datwt,sou,imgar[iexp],nthrds,wverb)
      else:
        # Extended imaging
        ssf.migoffallw(datwt,sou,nhy,nhx,sym,imgar[iexp],nthrds,wverb)
      # Increase number of traces
      ntr += self.__nrec[iexp]

    # Sum over all partial images
    img = np.sum(imgar,axis=0)

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
    datt = np.transpose(dat,(0,2,3,1)) # [nexp,nwc,ny,nx] -> [nexp,ny,nx,nwc]
    # Pad to the original frequency range
    padb = int(ow/dw); pade = nw - nwc - padb
    dattpad  = np.pad(datt,((0,0),(0,0),(0,0),(padb,pade)),mode='constant')  # [*,nwc] -> [*,nw]
    # Pad for the inverse FFT
    dattpadp = np.pad(dattpad,((0,0),(0,0),(0,0),(0,nt-nw)),mode='constant') # [*,nw] -> [*,nt]
    # Inverse FFT and window to t0 (wavelet shift)
    datf2t = np.real(np.fft.ifft(dattpadp))
    if(it0 is not None):
      datf2tw = datf2t[:,:,:,it0:]
    else:
      datf2tw = datf2t
    # Pad and transpose
    datf2tp = np.pad(datf2tw,((0,0),(0,0),(0,0),(0,n1-(nt-it0))),mode='constant')

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

