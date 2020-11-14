"""
Imaging/modeling data based on source
and receiver coordinates
@author: Joseph Jennings
@version: 2020.08.10
"""
import numpy as np
from oway.ssr3 import ssr3, interp_slow
from oway.utils import fft1, ifft1, make_sht_cube
from scaas.off2ang import off2angssk,off2angkzx
from genutils.ptyprint import progressbar
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
      print("Warning: Some source coordinates are greater than model size")
    if(np.any(srcxs < ox) or np.any(srcys <  oy)):
      print("Warning: Some source coordinates are less than model size")
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
      print("Warning: Some receiver coordinates are greater than model size")
    if(np.any(recxs < ox) or np.any(recys <  oy)):
      print("Warning: Some receiver coordinates are less than model size")
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

    # Subsurface offsets
    self.__rnhx = None; self.__ohx = None; self.__dhx = None
    self.__rnhy = None; self.__ohy = None; self.__dhy = None

    # Angle
    self.__na = None; self.__oa = None; self.__da = None

  def get_freq_axis(self):
    """ Returns the frequency axis """
    return self.__nwc,self.__ow,self.__dw

  def model_data(self,wav,dt,t0,minf,maxf,vel,ref,jf=1,nrmax=3,eps=0.,dtmax=5e-05,time=True,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False):
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
      eps    - stability parameter [0.]
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
    self.__nwo,self.__ow,self.__dw,wfft = fft1(wav,dt,minf=minf,maxf=maxf)
    wfftd = wfft[::jf]
    self.__nwc = wfftd.shape[0] # Get the number of frequencies to compute
    self.__dwc = jf*self.__dw

    if(sverb or wverb): print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz ,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz ,     # Spatial Samplings
               self.__nwc,self.__ow,self.__dwc,eps, # Frequency axis
               ntx,nty,px,py,                       # Taper and padding
               dtmax,nrmax,nthrds)                  # Reference velocities and threads

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
    for iexp in progressbar(range(self.__nexp),"nexp:",verb=sverb):
      # Get the source coordinates
      sy = self.__srcys[iexp]; sx = self.__srcxs[iexp]
      isy = int((sy-self.__oy)/self.__dy+0.5); isx = int((sx-self.__ox)/self.__dx+0.5)
      # Create the source for this shot
      sou[:] = 0.0
      sou[:,isy,isx]  = wfftd[:]
      # Downward continuation
      datw[:] = 0.0
      ssf.modallw(ref,sou,datw,wverb)
      #plt.figure()
      #plt.imshow(np.real(datw[:,0,:]),cmap='gray',interpolation='sinc',aspect='auto')
      #plt.show()
      # Restrict to receiver locations
      datwt = np.ascontiguousarray(np.transpose(datw,(1,2,0)))  # [nwc,ny,nx] -> [ny,nx,nwc]
      ssf.restrict_data(self.__nrec[iexp],self.__recys[ntr:],self.__recxs[ntr:],self.__oy,self.__ox,datwt,recw[ntr:,:])
      # Increase number of traces
      ntr += self.__nrec[iexp]

    if(time):
      rect = ifft1(recw,self.__nwo,self.__ow,self.__dw,nt,it0)
      return rect
    else:
      return recw

  def image_data(self,dat,dt,minf,maxf,vel,jf=1,nhx=0,nhy=0,sym=True,nrmax=3,eps=0.,dtmax=5e-05,wav=None,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False):
    """
    3D migration of shot profile data via the one-way wave equation (single-square
    root split-step fourier method). Input data are assumed to follow
    the default geometry (sources and receivers on a regular grid)

    Parameters:
      dat     - input shot profile data [ntr,nt]
      dt      - temporal sampling of input data
      minf    - minimum frequency to image in the data [Hz]
      maxf    - maximum frequency to image in the data [Hz]
      vel     - input migration velocity model [nz,ny,nx]
      jf      - frequency decimation factor [1]
      nhx     - number of subsurface offsets in x to compute [0]
      nhy     - number of subsurface offsets in y to compute [0]
      sym     - symmetrize the subsurface offsets [True]
      nrmax   - maximum number of reference velocities [3]
      eps     - stability parameter [0.]
      dtmax   - maximum time error [5e-05]
      wav     - input wavelet [None,assumes an impulse at zero lag]
      ntx     - size of taper in x direction [0]
      nty     - size of taper in y direction [0]
      px      - amount of padding in x direction (samples) [0]
      py      - amount of padding in y direction (samples) [0]
      nthrds  - number of OpenMP threads for frequency parallelization [1]
      sverb   - verbosity flag for shot progress bar [True]
      wverb   - verbosity flag for frequency progress bar [False]

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
    self.__nwo,self.__ow,self.__dw,wfft = fft1(wav,dt,minf=minf,maxf=maxf)
    wfftd = wfft[::jf]
    self.__nwc = wfftd.shape[0] # Get the number of frequencies for imaging
    self.__dwc = self.__dw*jf

    if(sverb or wverb): print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))

    # Create frequency domain data
    _,_,_,dfft = fft1(dat,dt,minf=minf,maxf=maxf)
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
               dtmax,nrmax,nthrds)                  # Reference velocities and threads

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    # Allocate partial image array
    if(nhx == 0 and nhy == 0):
      imgtmp = np.zeros([self.__nz,self.__ny,self.__nx],dtype='float32')
      oimg   = np.zeros([self.__nz,self.__ny,self.__nx],dtype='float32')
    else:
      if(sym):
        # Create axes
        self.__rnhx = 2*nhx+1; self.__ohx = -nhx*self.__dx; self.__dhx = self.__dx
        self.__rnhy = 2*nhy+1; self.__ohy = -nhy*self.__dy; self.__dhy = self.__dy
        imgtmp = np.zeros([self.__rnhy,self.__rnhx,self.__nz,self.__ny,self.__nx],dtype='float32')
        oimg   = np.zeros([self.__rnhy,self.__rnhx,self.__nz,self.__ny,self.__nx],dtype='float32')
      else:
        # Create axes
        self.__rnhx = nhx+1; self.__ohx = 0; self.__dhx = self.__dx
        self.__rnhy = nhy+1; self.__ohy = 0; self.__dhy = self.__dy
        imgtmp = np.zeros([self.__rnhy,self.__rnhx,self.__nz,self.__ny,self.__nx],dtype='float32')
        oimg   = np.zeros([self.__rnhy,self.__rnhx,self.__nz,self.__ny,self.__nx],dtype='float32')
      # Allocate memory necessary for extension
      ssf.set_ext(nhy,nhx,sym)

    # Loop over sources
    ntr = 0
    for iexp in progressbar(range(self.__nexp),"nexp:",verb=sverb):
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
      # Initialize temporary image
      imgtmp[:] = 0.0
      if(nhx == 0 and nhy == 0):
        # Conventional imaging
        ssf.migallw(datwt,sou,imgtmp,wverb)
      else:
        # Extended imaging
        ssf.migoffallw(datwt,sou,imgtmp,wverb)
      oimg += imgtmp
      # Increase number of traces
      ntr += self.__nrec[iexp]

    # Free memory for extension
    if(nhx != 0 or nhy != 0):
      ssf.del_ext()

    return oimg

  def get_off_axis(self):
    """ Returns the x subsurface offset extension axis """
    if(self.__rnhx is None):
      raise Exception("Cannot return x subsurface offset axis without running extended imaging")
    return self.__rnhx, self.__ohx, self.__dhx

  def fwemva(self,dslo,dat,dt,minf,maxf,vel,jf=1,nrmax=3,eps=0.,dtmax=5e-05,wav=None,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False):
    """
    3D Forward WEMVA operator

    Parameters:
      dat     - input shot profile data [ntr,nt]
      dt      - temporal sampling of input data
      minf    - minimum frequency to image in the data [Hz]
      maxf    - maximum frequency to image in the data [Hz]
      vel     - input migration velocity model [nz,ny,nx]
      jf      - frequency decimation factor [1]
      nrmax   - maximum number of reference velocities [3]
      eps     - stability parameter [0.]
      dtmax   - maximum time error [5e-05]
      wav     - input wavelet [None,assumes an impulse at zero lag]
      ntx     - size of taper in x direction [0]
      nty     - size of taper in y direction [0]
      px      - amount of padding in x direction (samples) [0]
      py      - amount of padding in y direction (samples) [0]
      nthrds  - number of OpenMP threads for frequency parallelization [1]
      sverb   - verbosity flag for shot progress bar [True]
      wverb   - verbosity flag for frequency progress bar [False]

    Returns:
      a linearized image perturbation (forward wemva applied to slowness) [nz,ny,nx]
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
    self.__nwo,self.__ow,self.__dw,wfft = fft1(wav,dt,minf=minf,maxf=maxf)
    wfftd = wfft[::jf]
    self.__nwc = wfftd.shape[0] # Get the number of frequencies for imaging
    self.__dwc = self.__dw*jf

    if(sverb or wverb): print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))

    # Create frequency domain data
    _,_,_,dfft = fft1(dat,dt,minf=minf,maxf=maxf)
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
               dtmax,nrmax,nthrds)                  # Reference velocities and threads

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    # Allocate temporary partial image
    dimgtmp = np.zeros([self.__nz,self.__ny,self.__nx],dtype='complex64')
    odimg   = np.zeros([self.__nz,self.__ny,self.__nx],dtype='complex64')

    # Loop over sources
    ntr = 0
    for iexp in progressbar(range(self.__nexp),"nexp:",verb=sverb):
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
      # Initialize temporary image
      dimgtmp[:] = 0.0
      # Forward WEMVA
      ssf.fwemvaallw(sou,datwt,dslo,dimgtmp,verb=wverb)
      odimg += dimgtmp
      # Increase number of traces
      ntr += self.__nrec[iexp]

    return np.real(odimg)

  def awemva(self,dimg,dat,dt,minf,maxf,vel,jf=1,nrmax=3,eps=0.,dtmax=5e-05,wav=None,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False):
    """
    3D Adjoint WEMVA operator

    Parameters:
      dat     - input shot profile data [ntr,nt]
      dt      - temporal sampling of input data
      minf    - minimum frequency to image in the data [Hz]
      maxf    - maximum frequency to image in the data [Hz]
      vel     - input migration velocity model [nz,ny,nx]
      jf      - frequency decimation factor [1]
      nrmax   - maximum number of reference velocities [3]
      eps     - stability parameter [0.]
      dtmax   - maximum time error [5e-05]
      wav     - input wavelet [None,assumes an impulse at zero lag]
      ntx     - size of taper in x direction [0]
      nty     - size of taper in y direction [0]
      px      - amount of padding in x direction (samples) [0]
      py      - amount of padding in y direction (samples) [0]
      nthrds  - number of OpenMP threads for frequency parallelization [1]
      sverb   - verbosity flag for shot progress bar [True]
      wverb   - verbosity flag for frequency progress bar [False]

    Returns:
      a slowness perturbation (adjoint wemva applied to image perturbation) [nz,ny,nx]
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
    self.__nwo,self.__ow,self.__dw,wfft = fft1(wav,dt,minf=minf,maxf=maxf)
    wfftd = wfft[::jf]
    self.__nwc = wfftd.shape[0] # Get the number of frequencies for imaging
    self.__dwc = self.__dw*jf

    if(sverb or wverb): print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))

    # Create frequency domain data
    _,_,_,dfft = fft1(dat,dt,minf=minf,maxf=maxf)
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
               dtmax,nrmax,nthrds)                  # Reference velocities and threads

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    # Allocate temporary partial image
    dslotmp = np.zeros([self.__nz,self.__ny,self.__nx],dtype='complex64')
    odslo   = np.zeros([self.__nz,self.__ny,self.__nx],dtype='complex64')

    # Loop over sources
    ntr = 0
    for iexp in progressbar(range(self.__nexp),"nexp:",verb=sverb):
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
      # Initialize temporary image
      dslotmp[:] = 0.0
      # Adjoint WEMVA
      ssf.awemvaallw(sou,datwt,dslotmp,dimg,verb=wverb)
      odslo += dslotmp
      # Increase number of traces
      ntr += self.__nrec[iexp]

    return np.real(odslo)

  def to_angle(self,img,mode='kzx',amax=None,na=None,nthrds=4,transp=False,
               eps=1.0,oro=None,dro=None,verb=False):
    """
    Converts the subsurface offset gathers to opening angle gathers

    Parameters
      img    - Image extended over subsurface offsets [nhy,nhx,nz,ny,nx]
      mode   - mode of computing angle gathers [kzx/ssk]
      amax   - Maximum angle over which to compute angle gathers [60/70]
      na     - Number of angles on the angle axis [nhx/281]
      nthrds - Number of OpenMP threads to use (parallelize over image point axis) [4]
      transp - Transpose the output to have shape [nx,na,nz]
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
    else:
      raise Exception("Mode %s not recognized. Available modes are 'kzx' or 'ssk'"%(mode))

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

