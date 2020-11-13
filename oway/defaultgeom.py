"""
Default geometry for synthetics
Sources and receivers on the surface
and distributed evenly across the surface
@author: Joseph Jennings
@version: 2020.11.12
"""
import numpy as np
from oway.ssr3 import ssr3, interp_slow
from scaas.off2ang import off2angssk
from genutils.ptyprint import progressbar
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
    # Build source coordinates
    self.__scoords = []
    for isy in range(nsy):
      sy = int(osy + isy*dsy)
      for isx in range(nsx):
        sx = int(osx + isx*dsx)
        self.__scoords.append([sy,sx])
    self.__nexp = len(self.__scoords)

    # Frequency axis
    self.__nwo = None; self.__ow = None; self.__dw = None;

    # Subsurface offsets
    self.__rnhx = None; self.__ohx = None; self.__dhx = None
    self.__rnhy = None; self.__ohy = None; self.__dhy = None

    # Angle
    self.__na = None; self.__oa = None; self.__da = None

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

  def model_data(self,wav,dt,t0,minf,maxf,vel,ref,jf=1,nrmax=3,eps=0.,dtmax=5e-05,time=True,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False) -> np.ndarray:
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
      jf     - frequency decimation factor [1]
      nrmax  - maximum number of reference velocities [3]
      eps    - stability parameter [0.]
      dtmax  - maximum time error [5e-05]
      time   - return the data back in the time domain [True]
      ntx    - size of taper in x direction (samples) [0]
      nty    - size of taper in y direction (samples) [0]
      px     - amount of padding in x direction (samples)
      py     - amount of padding in y direction (samples)
      nthrds - number of OpenMP threads for parallelizing over frequency [1]
      sverb  - verbosity flag for shot progress bar [True]
      wverb  - verbosity flag for frequency progressbar [False]

    Returns the data at the surface (in time or frequency) [nw,nry,nrx]
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
               dtmax,nrmax,nthrds)                  # Reference velocities

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    # Allocate output data (surface wavefield)
    datw = np.zeros([self.__nexp,self.__nwc,self.__ny,self.__nx],dtype='complex64')

    # Allocate the source for one shot
    sou = np.zeros([self.__nwc,self.__ny,self.__nx],dtype='complex64')

    # Loop over sources
    k = 0
    for icrd in progressbar(self.__scoords,"nexp:",verb=sverb):
      # Get the source coordinates
      sy = icrd[0]; sx = icrd[1]
      # Create the source for this shot
      sou[:] = 0.0
      sou[:,sy,sx]  = wfftd[:]
      # Downward continuation
      ssf.modallw(ref,sou,datw[k],wverb)
      k += 1

    # Reshape output data
    datwr = datw.reshape([self.__nsy,self.__nsx,self.__nwc,self.__ny,self.__nx])

    if(time):
      # Inverse fourier transform
      datt = self.data_f2t(datwr,self.__nwo,self.__ow,self.__dwc,nt,it0)
      return datt
    else:
      return datwr

  def image_data(self,dat,dt,minf,maxf,vel,jf=1,nhx=0,nhy=0,sym=True,nrmax=3,eps=0.0,dtmax=5e-05,wav=None,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False) -> np.ndarray:
    """
    3D migration of shot profile data via the one-way wave equation (single-square
    root split-step fourier method). Input data are assumed to follow
    the default geometry (sources and receivers on a regular grid)

    Parameters:
      dat    - input shot profile data [nsy,nsx,nry,nrx,nt]
      dt     - temporal sampling of input data
      minf   - minimum frequency to image in the data [Hz]
      maxf   - maximum frequency to image in the data [Hz]
      vel    - input migration velocity model [nz,ny,nx]
      jf     - frequency decimation factor
      nhx    - number of subsurface offsets in x to compute [0]
      nhy    - number of subsurface offsets in y to compute [0]
      sym    - symmetrize the subsurface offsets [True]
      nrmax  - maximum number of reference velocities [3]
      eps    - stability parameter [0.]
      dtmax  - maximum time error [5e-05]
      wav    - input wavelet [None,assumes an impulse at zero lag]
      ntx    - size of taper in x direction [0]
      nty    - size of taper in y direction [0]
      px     - amount of padding in x direction (samples) [0]
      py     - amount of padding in y direction (samples) [0]
      nthrds - number of OpenMP threads for parallelizing over frequency [1]
      sverb  - verbosity flag for shot progress bar [True]
      wverb  - verbosity flag for frequency progress bar [False]

    Returns:
      an image created from the data [nhy,nhx,nz,ny,nx]
    """
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
    datt = np.transpose(dfftd,(0,1,4,2,3)) # [nsy,nsx,ny,nx,nwc] -> [nsy,nsx,nwc,ny,nx]
    datw = np.ascontiguousarray(datt.reshape([self.__nexp,self.__nwc,self.__ny,self.__nx]))

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz ,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz ,     # Spatial Samplings
               self.__nwc,self.__ow,self.__dwc,eps, # Frequency axis
               ntx,nty,px,py,                       # Taper and padding
               dtmax,nrmax,nthrds)                  # Reference velocities

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    # Allocate partial image array
    if(nhx == 0 and nhy == 0):
      imgar = np.zeros([self.__nexp,self.__nz,self.__ny,self.__nx],dtype='float32')
    else:
      if(sym):
        # Create axes
        self.__rnhx = 2*nhx+1; self.__ohx = -nhx*self.__dx; self.__dhx = self.__dx
        self.__rnhy = 2*nhy+1; self.__ohy = -nhy*self.__dy; self.__dhy = self.__dy
        # Allocate image array
        imgar = np.zeros([self.__nexp,self.__rnhy,self.__rnhx,self.__nz,self.__ny,self.__nx],dtype='float32')
      else:
        # Create axes
        self.__rnhx = nhx+1; self.__ohx = 0; self.__dhx = self.__dx
        self.__rnhy = nhy+1; self.__ohy = 0; self.__dhy = self.__dy
        # Allocate image array
        imgar = np.zeros([self.__nexp,self.__rnhy,self.__rnhx,self.__nz,self.__ny,self.__nx],dtype='float32')
      # Allocate memory necessary for extension
      ssf.set_ext(nhy,nhx,sym)

    # Allocate the source for one shot
    sou = np.zeros([self.__nwc,self.__ny,self.__nx],dtype='complex64')

    # Loop over sources
    k = 0
    for icrd in progressbar(self.__scoords,"nexp:",verb=sverb):
      # Get the source coordinates
      sy = icrd[0]; sx = icrd[1]
      # Create the source for this shot
      sou[:] = 0.0
      sou[:,sy,sx]  = wfftd[:]
      if(nhx == 0 and nhy == 0):
        # Conventional imaging
        ssf.migallw(datw[k],sou,imgar[k],wverb)
      else:
        # Extended imaging
        ssf.migoffallw(datw[k],sou,imgar[k],wverb)
      k += 1

    # Sum over all partial images
    img = np.sum(imgar,axis=0)

    # Free memory for extension
    if(nhx != 0 or nhy != 0):
      ssf.del_ext()

    return img

  def get_off_axis(self):
    """ Returns the x subsurface offset extension axis """
    if(self.__rnhx is None):
      raise Exception("Cannot return x subsurface offset axis without running extended imaging")
    return self.__rnhx, self.__ohx, self.__dhx

  def fwemva(self,dslo,dat,dt,minf,maxf,vel,jf=1,nrmax=3,eps=0.0,dtmax=5e-05,wav=None,
             ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False) -> np.ndarray:
    """
    Applies the forward WEMVA operator

    Parameters:
      dslo   - the input slowness perturbation
      dat    - the input data
      dt     - temporal sampling interval
      minf   - minimum frequency to image in the data [Hz]
      maxf   - maximim frequency to image in the data [Hz]
      vel    - input migration velocity model [nz,ny,nx]
      jf     - frequency decimation factor [1]
      nrmax  - maximum number of reference velocities [3]
      eps    - stability parameter [0.]
      dtmax  - maximum time error [5e-05]
      wav    - input wavelet [None,assumes an impulse at zero lag]
      ntx    - size of taper in x direction [0]
      nty    - size of taper in y direction [0]
      px     - amount of padding in x direction (samples) [0]
      py     - amount of padding in y direction (samples) [0]
      nthrds - number of OpenMP threads for parallelizing over frequency [1]
      sverb  - verbosity flag for shot progress bar [True]
      wverb  - verbosity flag for frequency progress bar [False]

    Returns:
      a linearized image perturbation (forward wemva applied to slowness) [nz,ny,nx]
    """
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
    datt = np.transpose(dfftd,(0,1,4,2,3)) # [nsy,nsx,ny,nx,nwc] -> [nsy,nsx,nwc,ny,nx]
    datw = np.ascontiguousarray(datt.reshape([self.__nexp,self.__nwc,self.__ny,self.__nx]))

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz ,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz ,     # Spatial Samplings
               self.__nwc,self.__ow,self.__dwc,eps, # Frequency axis
               ntx,nty,px,py,                       # Taper and padding
               dtmax,nrmax,nthrds)                  # Reference velocities

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    dimgar = np.zeros([self.__nexp,self.__nz,self.__ny,self.__nx],dtype='complex64')

    # Allocate the source for one shot
    sou = np.zeros([self.__nwc,self.__ny,self.__nx],dtype='complex64')

    # Loop over sources
    k = 0
    for icrd in progressbar(self.__scoords,"nexp:",verb=sverb):
      # Get the source coordinates
      sy = icrd[0]; sx = icrd[1]
      # Create the source for this shot
      sou[:] = 0.0
      sou[:,sy,sx]  = wfftd[:]
      ssf.fwemvaallw(sou,datw[k],dslo,dimgar[k],verb=wverb)
      k += 1

    # Sum over all partial images
    dimg = np.sum(dimgar,axis=0)

    return np.real(dimg)

  def awemva(self,dimg,dat,dt,minf,maxf,vel,jf=1,nrmax=3,eps=0.0,dtmax=5e-05,wav=None,
             ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False) -> np.ndarray:
    """
    Applies the forward WEMVA operator

    Parameters:
      dimg   - the input image perturbation
      dat    - the input data
      dt     - temporal sampling interval
      minf   - minimum frequency to image in the data [Hz]
      maxf   - maximim frequency to image in the data [Hz]
      vel    - input migration velocity model [nz,ny,nx]
      jf     - frequency decimation factor [1]
      nrmax  - maximum number of reference velocities [3]
      eps    - stability parameter [0.]
      dtmax  - maximum time error [5e-05]
      wav    - input wavelet [None,assumes an impulse at zero lag]
      ntx    - size of taper in x direction [0]
      nty    - size of taper in y direction [0]
      px     - amount of padding in x direction (samples) [0]
      py     - amount of padding in y direction (samples) [0]
      nthrds - number of OpenMP threads for parallelizing over frequency [1]
      sverb  - verbosity flag for shot progress bar [True]
      wverb  - verbosity flag for frequency progress bar [False]

    Returns:
      a slowness perturbation (adjoint wemva applied to image) [nz,ny,nx]
    """
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
    datt = np.transpose(dfftd,(0,1,4,2,3)) # [nsy,nsx,ny,nx,nwc] -> [nsy,nsx,nwc,ny,nx]
    datw = np.ascontiguousarray(datt.reshape([self.__nexp,self.__nwc,self.__ny,self.__nx]))

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz ,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz ,     # Spatial Samplings
               self.__nwc,self.__ow,self.__dwc,eps, # Frequency axis
               ntx,nty,px,py,                       # Taper and padding
               dtmax,nrmax,nthrds)                  # Reference velocities

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    dsloar = np.zeros([self.__nexp,self.__nz,self.__ny,self.__nx],dtype='complex64')

    # Allocate the source for one shot
    sou = np.zeros([self.__nwc,self.__ny,self.__nx],dtype='complex64')

    # Loop over sources
    k = 0
    for icrd in progressbar(self.__scoords,"nexp:",verb=sverb):
      # Get the source coordinates
      sy = icrd[0]; sx = icrd[1]
      # Create the source for this shot
      sou[:] = 0.0
      sou[:,sy,sx]  = wfftd[:]
      ssf.awemvaallw(sou,datw[k],dsloar[k],dimg,verb=wverb)
      k += 1

    # Sum over all partial images
    dslo = np.sum(dsloar,axis=0)

    return np.real(dslo)

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

  def to_angle(self,img,amax=70,na=281,nthrds=4,transp=False,oro=None,dro=None,verb=False):
    """
    Converts the subsurface offset gathers to opening angle gathers

    Parameters
      img    - Image extended over subsurface offsets [nhy,nhx,nz,ny,nx]
      amax   - Maximum angle over which to compute angle gathers [70]
      na     - Number of angles on the angle axis [281]
      nthrds - Number of OpenMP threads to use (parallelize over image point axis) [4]
      transp - Transpose the output to have shape [na,nx,nz]
      verb   - Verbosity flag [False]

    Returns the angle gathers [nro,nx,na,nz]
    """
    if(mode == 'kzx'):
      if(amax is None): amax = 60
      if(na is None): na = self.__rnhx
      # Handle the case of residual migration input
      itransp = False
      if(len(img.shape) == 4): itransp = True
      # Compute angle axis
      self.__na = na; self.__oa = -amax; self.__da = 2*amax/na
      angs = off2angkzx(img,self.__ohx,self.__dhx,self.__dz,na=na,amax=amax,transp=itransp,cverb=verb)
      if(transp):
        # [naz,na,nz,ny,nx] -> [ny,nx,naz,na,nz]
        angst = np.ascontiguousarray(np.transpose(angs,(3,4,0,1,2)))
      else:
        angst = angs
      return angst
    elif(mode == 'ssk'):
      if(amax is None): amax = 70
      if(na is None): na = 281
      # Assume ny = 1
      imgin = img[0,:,:,0,:]
      amin = -amax; avals = np.linspace(amin,amax,na)
      # Compute angle axis
      self.__na = na; self.__da = avals[1] - avals[0]; self.__oa = avals[0]
      return off2angssk(imgin,self.__ohx,self.__dhx,self.__dz,na=na,amax=amax,nta=601,ota=-3,dta=0.01,
                      nthrds=nthrds,transp=transp,oro=oro,dro=dro,verb=verb)

  def get_ang_axis(self):
    """ Returns the opening angle extension axis """
    return self.__na, self.__oa, self.__da

  def test_freq_axis(self,n1,dt,minf,maxf,jf=1):
    """
    For testing different frequency axes based on
    the input wavelet time axis

    Parameters:
      n1   - length of wavelet
      dt   - temporal sampling of wavelet
      minf - minimum frequency to propagate
      maxf - maximum frequency to propagate
      jf   - frequency decimation factor [1]

    Returns:
      Nothing. Just a verbose output of the frequency axis
    """
    nt = 2*self.next_fast_size(int((n1+1)/2))
    if(nt%2): nt += 1
    nw = int(nt/2+1)
    dw = 1/(nt*dt)
    # Min and max frequencies
    begw = int(minf/dw); endw = int(maxf/dw)
    nwc = (endw-begw)/jf
    print("Test frequency axis: nw=%d ow=%d dw=%f"%(nwc,minf,dw*jf))

  def get_scoords(self):
    """ Returns the source injection coordinates """
    return self.__scoords

  def plot_acq(self,mod=None,show=True,**kwargs):
    """ Plots the acquisition on the slowness model """
    # Plot the slowness model
    if(mod is None):
      mod = np.zeros([self.__nz,self.__ny,self.__nx],dtype='float32') + 2.5
    vmin = np.min(mod); vmax = np.max(mod)
    if(self.__ny == 1):
      # 2D acquisition
      fig = plt.figure(figsize=(kwargs.get('wbox',14),kwargs.get('hbox',7)))
      ax = fig.gca()
      # Plot model
      im = ax.imshow(mod[:,0,:],extent=[0,self.__nx,self.__nz,0],vmin=kwargs.get('vmin',vmin),vmax=kwargs.get('vmax',vmax),
                     cmap=kwargs.get('cmap','jet'))
      ax.set_xlabel(kwargs.get('xlabel','X (gridpoints)'),fontsize=kwargs.get('labelsize',14))
      ax.set_ylabel(kwargs.get('ylabel','Z (gridpoints)'),fontsize=kwargs.get('labelsize',14))
      ax.set_title(kwargs.get('title',''),fontsize=kwargs.get('labelsize',14))
      ax.tick_params(labelsize=kwargs.get('labelsize',14))
      # Make receiver coords
      zplt = 5
      recx = np.linspace(0,self.__nx-1,self.__nx)
      rzros = np.zeros(self.__nx) + zplt
      # Make source coords
      scoordsn = np.asarray(self.__scoords)[:,1]
      szros    = np.zeros(len(scoordsn)) + zplt
      # Plot geometry
      #plt.scatter(recx,rzros,c='tab:green',marker='v')
      plt.scatter(scoordsn,szros,c='yellow',marker='*')
      if(show):
        plt.show()

