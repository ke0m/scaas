"""
Default geometry for synthetics
Sources and receivers on the surface
and distributed evenly across the surface.
This code is designed to be distributed across nodes in a cluster

@author: Joseph Jennings
@version: 2020.07.26
"""
import numpy as np
from oway.ssr3 import interp_slow
from oway.ssr3wrap import ssr3modshots, ssr3migshots, ssr3migoffshots
from scaas.off2ang import off2ang
import matplotlib.pyplot as plt

class defaultgeomnode:
  """
  Functions for modeling and imaging with a
  standard synthetic source receiver geometry
  """
  def __init__(self,nx,dx,ny,dy,nz,dz,                             # Model size
               nsx,dsx,nsy,dsy,osx=0.0,osy=0.0,                    # Source geometry
               nrx=None,drx=1.0,orx=0.0,nry=None,dry=1.0,ory=0.0,  # Receiver geometry
               ox=0.0,oy=0.0,oz=0.0):                              # Model origins

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
      ox    - origin of model/image x-axis
      oy    - origin of model/image y-axis
      oz    - origin of model/image z-axis

    Returns:
      a default geom node object
    """
    # Spatial axes
    self.__nx = nx; self.__ox = ox; self.__dx = dx
    self.__ny = ny; self.__oy = oy; self.__dy = dy
    self.__nz = nz; self.__oz = oz; self.__dz = dz

    # Source gometry
    osx *= dx; dsx *= dx
    osy *= dy; dsy *= dy
    # Number of sources per shot
    self.__nexp = nsy*nsx
    self.__nsrc = np.ones(self.__nexp,dtype='int32')
    # Total number of sources
    self.__nwav = np.sum(self.__nsrc)
    # Build source coordinates
    self.__srcy = np.zeros(self.__nexp,dtype='float32')
    self.__srcx = np.zeros(self.__nexp,dtype='float32')
    k = 0
    for isy in range(nsy):
      self.__srcy[k] = osy + isy*dsy
      for isx in range(nsx):
        self.__srcx[k] = osx + isx*dsx
        k += 1

    # Receiver geometry
    if(nry is None): nry = ny
    if(nrx is None): nrx = nx
    orx *= dx; drx *= dx
    ory *= dy; dry *= dy
    # Number of receivers per shot
    self.__nrec = np.zeros(self.__nexp,dtype='int32') + nry*nrx
    # Total number of traces
    self.__ntr = np.sum(self.__nrec)
    # Build receiver coordinates
    self.__recy = np.zeros(self.__ntr,dtype='float32')
    self.__recx = np.zeros(self.__ntr,dtype='float32')
    k = 0
    #TODO: potentially use a mesh grid here
    for iexp in range(self.__nexp):
      for iry in range(nry):
        self.__recy[k] = ory + iry*dry
        for irx in range(nrx):
          self.__recx[k] = orx + irx*drx
          k += 1

    # Data frequency axis and imaging/modeling axis
    self.__nwo = None; self.__ow = None; self.__dw  = None
    self.__nwc = None;                   self.__dwc = None

    # Subsurface offsets
    self.__sym = True
    self.__nhx = 0; self.__rnhx = None; self.__ohx = None; self.__dhx = None
    self.__nhy = 0; self.__rnhy = None; self.__ohy = None; self.__dhy = None

    # Angle
    self.__na = None; self.__oa = None; self.__da = None

    # Tapering and padding
    self.__nty = 0; self.__ntx = 0
    self.__py  = 0; self.__px  = 0

    # Reference slownesses
    self.__nrmax = 3; self.__dtmax = 5e-05

    # Velocity and reflectivity
    self.__slo = None; self.__ref = None

    # Verbosity and threading
    self.__verb = 0; self.__nthrds = 1

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

  def create_mod_chunks(self,nchnks,wavs):
    """
    Creates chunks for distributed modeling over a cluster

    Parameters:
      nchnks - number of chunks to create
      dat    - input data [ntr,n1]
      wavs   - input wavelets [nwav,n1] or wavelet [n1]

    Returns a list of dictionary arguments for each chunk. [nchnks]
    """
    # Get wavelet dimensions
    n1 = wavs.shape[-1]
    if(wavs.ndim == 1):
      wav = np.repeat(wavs[np.newaxis,:],self.__nexp,axis=0)

    # Allocate the data
    dat = np.zeros([self.__ntr,n1],dtype='complex64')
    ntr = dat.shape[0]; nwav = wav.shape[0]
    ochnks = []
    expchnks = self.splitnum(self.__nexp,nchnks)

    k = 0
    begs = 0; ends = 0; begr = 0; endr = 0
    for ichnk in range(len(expchnks)):
      # Get data and sources for each chunk
      nsrccnk = np.zeros(expchnks[ichnk],dtype='int32')
      nreccnk = np.zeros(expchnks[ichnk],dtype='int32')
      for iexp in range(expchnks[ichnk]):
        nsrccnk[iexp] = self.__nsrc[k]
        nreccnk[iexp] = self.__nrec[k]
        ends += self.__nsrc[k]
        endr += self.__nrec[k]
        k += 1
      # Chunked source data
      sychnk  = self.__srcy[begs:ends]
      sxchnk  = self.__srcx[begs:ends]
      srcchnk = wav[begs:ends,:]
      # Chunked receiver data
      rychnk  = self.__recy[begr:endr]
      rxchnk  = self.__recx[begr:endr]
      datchnk = dat[begr:endr,:]
      # Update positions
      begs = ends; begr = endr
      # Put data in into dict
      idict = {}
      idict['srcy'] = sychnk; idict['srcx'] = sxchnk; idict['nsrc'] = nsrccnk
      idict['recy'] = rychnk; idict['recx'] = rxchnk; idict['nrec'] = nreccnk
      idict['wav']  = srcchnk; idict['dat'] = datchnk
      ochnks.append(idict)

    return ochnks

  def set_model_pars(self,vel,ref,nrmax=3,dtmax=5e-05,ntx=0,nty=0,px=0,py=0,wverb=False,everb=False,nthrds=1):
    """
    Sets the non-changing parameters (same for all chunks) for calling
    model chunk outside of the model_data function

    Parameters:
      vel    - modeling velocity [nz,ny,nx]
      ref    - reflectivity [nz,ny,nx]
      nrmax  - maximum number of reference velocities [3]
      dtmax  - maximum time error [5e-05]
      ntx    - length of taper in x-direction (samples) [0]
      nty    - length of taper in y-direction (samples) [0]
      px     - amount of padding in x-direction (samples) [0]
      py     - amount of padding in y-direction (samples) [0]
      everb  - shot progress bar [False]
      wverb  - frequency progress bar [False]
      nthrds - number of OpenMP threads to use [1]
    """
    # Slowness and reflectivity
    self.__slo = 1/vel
    self.__ref = ref

    # Reference velocities
    self.__nrmax = nrmax
    self.__dtmax = dtmax

    # Tapering and padding
    self.__ntx = ntx; self.__nty = nty
    self.__px  = px ; self.__py  = py

    # Verbosity
    if(everb):
      self.__verb = 1
    if(wverb):
      self.__verb = 2

    # Threading
    self.__nthrds = nthrds

  def model_chunk(self,chunk):
    """
    Models a chunk of data. Takes input chunked
    coordinates and source wavelets.
    These chunks are created from the create_data_chunks function.

    Note: you need to have run the set_props function before running

    Parameters:
      chunk - a chunk created from the create_mod_chunks function.
              This chunk is a dictionary with the following keys:
              'nsrc'  - number of sources for each shot in chunk
              'srcy'  - y-coordinates of source location for the chunk of shots
              'srcx'  - x-coordinates of source location for the chunk of shots
              'wav'   - a chunk of source wavelets [inexp,nw]
              'nrec'  - number of receivers for each shot in chunk [nsrc]
              'recy'  - y-coordinates of receiver location for the chunk of shots [ntr]
              'recx'  - x-coordinates of receiver location for the chunk of shots [ntr]
              'dat'   - output data computed for chunk of shots [ntr,nw]

    Returns:
      the data modeled for that chunk [ntr,n1]
    """
    # Get number of shots in the chunk
    if(len(chunk['nsrc']) != len(chunk['nrec'])):
      raise Exception("nsrc and nrec array must be same length")
    nexp = len(chunk['nrec'])

    if(self.__slo is None or self.__ref is None):
      raise Exception("Must set the slowness and reflectivity with the set_model_pars function")

    # Perform the modeling
    ssr3modshots(self.__nx, self.__ny, self.__nz ,            # Spatial sizes
                 self.__ox, self.__oy, self.__oz ,            # Spatial origins
                 self.__dx, self.__dy, self.__dz ,            # Spatial samplings
                 self.__nwc,self.__ow, self.__dwc,            # Temporal frequency
                 self.__ntx, self.__nty,                      # Tapering
                 self.__px, self.__py,                        # Padding
                 self.__dtmax, self.__nrmax,                  # Reference slowness
                 self.__slo,                                  # Medium slowness
                 nexp,                                        # Number of experiments
                 chunk['nsrc'], chunk['srcy'], chunk['srcx'], # Num srcs and coords per exp
                 chunk['nrec'], chunk['recy'], chunk['recx'], # Num recs and coords per exp
                 chunk['wav'], self.__ref, chunk['dat'],      # Wavelets,reflectivity, output data
                 self.__nthrds, self.__verb)                  # Threading and verbosity

    return chunk['dat']

  def make_sht_cube(self,dat):
    """ 
    Makes a regular cube of shots from the input traces.
    Assumes that the data are already sorted by common
    shot

    Note only works for 2D data at the moment

    Parameters:
      dat - input shot data [ntr,nt]

    Returns:
      regular shot cube [nsht,nrx,nt]
    """
    # Get data dimensions
    if(dat.ndim != 2): 
      raise Exception("Data must be of dimension [ntr,nt]")
    nt = dat.shape[1]

    # Get maximum number of receivers
    nrecxmax = np.max(self.__nrec)

    # Output shot array
    shots = np.zeros([self.__nexp,nrecxmax,nt],dtype='float32')

    # Loop over all sources
    ntr = 0 
    for iexp in range(self.__nexp):
      shots[iexp,:self.__nrec[iexp],:] = dat[ntr:ntr+self.__nrec[iexp],:]
      ntr += self.__nrec[iexp]

    return shots

  def model_data(self,wav,dt,t0,minf,maxf,vel,ref,jf=1,nrmax=3,dtmax=5e-05,time=True,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False,client=None):
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

    if(sverb or wverb):
      print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))
      if(sverb):
        verb = 1
      if(wverb):
        verb = 2

    # Compute slowness
    slo = 1/vel

    # Output data array
    dat = np.zeros([self.__ntr,self.__nwc],dtype='complex64')

    if(time):
      # Inverse fourier transform
      datt = self.data_f2t(datwr,self.__nwo,self.__ow,self.__dwc,nt,it0)
      return datt
    else:
      return datwr

  def create_img_chunks(self,nchnks,wavs,dat):
    """
    Creates chunks for distributed imaging over a cluster

    Parameters:
      nchnks - number of chunks to create
      dat    - input data [ntr,n1]
      wavs   - input wavelets [nwav,n1] or [n1]

    Returns a list of dictionary arguments for each chunk. [nchnks]
    """
    # Check if data are complex
    if(dat.dtype != 'complex64'):
      raise Exception("Data must have been FFT'd before chunking")

    # Get wavelet dimensions
    n1 = wavs.shape[-1]
    if(wavs.ndim == 1):
      wav = np.repeat(wavs[np.newaxis,:],self.__nexp,axis=0)

    # Get data dimensions
    ntr = dat.shape[0]; nwav = wav.shape[0]
    ochnks = []
    expchnks = self.splitnum(self.__nexp,nchnks)

    k = 0
    begs = 0; ends = 0; begr = 0; endr = 0
    for ichnk in range(len(expchnks)):
      # Get data and sources for each chunk
      nsrccnk = np.zeros(expchnks[ichnk],dtype='int32')
      nreccnk = np.zeros(expchnks[ichnk],dtype='int32')
      for iexp in range(expchnks[ichnk]):
        nsrccnk[iexp] = self.__nsrc[k]
        nreccnk[iexp] = self.__nrec[k]
        ends += self.__nsrc[k]
        endr += self.__nrec[k]
        k += 1
      # Chunked source data
      sychnk  = self.__srcy[begs:ends]
      sxchnk  = self.__srcx[begs:ends]
      srcchnk = wav[begs:ends,:]
      # Chunked receiver data
      rychnk  = self.__recy[begr:endr]
      rxchnk  = self.__recx[begr:endr]
      datchnk = dat[begr:endr,:]
      # Update positions
      begs = ends; begr = endr
      # Put data in into dict
      idict = {}
      idict['srcy'] = sychnk; idict['srcx'] = sxchnk; idict['nsrc'] = nsrccnk
      idict['recy'] = rychnk; idict['recx'] = rxchnk; idict['nrec'] = nreccnk
      idict['wav'] = srcchnk; idict['dat'] = datchnk
      ochnks.append(idict)

    return ochnks

  def set_image_pars(self,vel,nhx=0,nhy=0,sym=True,nrmax=3,dtmax=5e-05,ntx=0,nty=0,px=0,py=0,
                     wverb=False,everb=False,nthrds=1):
    """
    Sets the non-changing parameters (same for all chunks) for calling
    image chunk outside of the image_data function

    Parameters:
      vel    - modeling velocity [nz,ny,nx]
      nhx    - number of x subsurface offsets [0]
      nhy    - number of y subsurface offsets [0]
      nrmax  - maximum number of reference velocities [3]
      dtmax  - maximum time error [5e-05]
      ntx    - length of taper in x-direction (samples) [0]
      nty    - length of taper in y-direction (samples) [0]
      px     - amount of padding in x-direction (samples) [0]
      py     - amount of padding in y-direction (samples) [0]
      everb  - shot progress bar [False]
      wverb  - frequency progress bar [False]
      nthrds - number of OpenMP threads to use [1]
    """
    # Slowness and reflectivity
    self.__slo = 1/vel

    # Subsurface offsets
    self.__nhx = nhx; self.__nhy = nhy; self.__sym = sym
    if(sym):
      # Create axes
      self.__rnhx = 2*nhx+1; self.__ohx = -nhx*self.__dx; self.__dhx = self.__dx
      self.__rnhy = 2*nhy+1; self.__ohy = -nhy*self.__dy; self.__dhy = self.__dy
    else:
      # Create axes
      self.__rnhx = nhx+1; self.__ohx = 0; self.__dhx = self.__dx
      self.__rnhy = nhy+1; self.__ohy = 0; self.__dhy = self.__dy

    # Reference velocities
    self.__nrmax = nrmax
    self.__dtmax = dtmax

    # Tapering and padding
    self.__ntx = ntx; self.__nty = nty
    self.__px  = px ; self.__py  = py

    # Verbosity
    if(everb):
      self.__verb = 1
    if(wverb):
      self.__verb = 2

    # Threading
    self.__nthrds = nthrds

  def image_chunk(self,chunk):
    """
    Models a chunk of data. Takes input chunked
    coordinates and source wavelets.
    These chunks are created from the create_data_chunks function.

    Note: you need to have run the set_props function before running

    Parameters:
      nsrc  - number of sources for each shot in chunk
      srcys - y-coordinates of source location for the chunk of shots
      srcxs - x-coordinates of source location for the chunk of shots
      wavs  - a chunk of source wavelets [inexp,nw]
      nrec  - number of receivers for each shot in chunk [nsrc]
      recys - y-coordinates of receiver location for the chunk of shots [ntr]
      recxs - x-coordinates of receiver location for the chunk of shots [ntr]
      dats  - input chunked data [ntr,nw]
    """
    # Get number of shots in the chunk
    if(len(chunk['nsrc']) != len(chunk['nrec'])):
      raise Exception("nsrc and nrec array must be same length")
    nexp = len(chunk['nrec'])

    if(self.__slo is None):
      raise Exception("Must set the slowness with the set_image_pars function")

    # Extended imaging
    if(self.__nhx != 0 or self.__nhy != 0):
      # Allocate the output image
      img = np.zeros([self.__rnhy,self.__rnhx,self.__nz,self.__ny,self.__nx],dtype='float32')

      # Perform the imaging
      ssr3migoffshots(self.__nx, self.__ny, self.__nz ,            # Spatial sizes
                      self.__ox, self.__oy, self.__oz ,            # Spatial origins
                      self.__dx, self.__dy, self.__dz ,            # Spatial samplings
                      self.__nwc,self.__ow, self.__dwc,            # Temporal frequency
                      self.__ntx, self.__nty,                      # Tapering
                      self.__px, self.__py,                        # Padding
                      self.__dtmax, self.__nrmax,                  # Reference slowness
                      self.__slo,                                  # Medium slowness
                      nexp,                                        # Number of experiments
                      chunk['nsrc'], chunk['srcy'], chunk['srcx'], # Num srcs and coords per exp
                      chunk['nrec'], chunk['recy'], chunk['recx'], # Num recs and coords per exp
                      chunk['dat'] , chunk['wav'],                 # Data and wavelets
                      self.__nhy, self.__nhx, self.__sym, img,     # Subsurface offsets and image
                      self.__nthrds, self.__verb)                  # Threading and verbosity
    else:
      # Allocate the output image
      img = np.zeros([self.__nz,self.__ny,self.__nx],dtype='float32')

      # Perform the imaging
      ssr3migshots(self.__nx, self.__ny, self.__nz ,            # Spatial sizes
                   self.__ox, self.__oy, self.__oz ,            # Spatial origins
                   self.__dx, self.__dy, self.__dz ,            # Spatial samplings
                   self.__nwc,self.__ow, self.__dwc,            # Temporal frequency
                   self.__ntx, self.__nty,                      # Tapering
                   self.__px, self.__py,                        # Padding
                   self.__dtmax, self.__nrmax,                  # Reference slowness
                   self.__slo,                                  # Medium slowness
                   nexp,                                        # Number of experiments
                   chunk['nsrc'], chunk['srcy'], chunk['srcx'], # Num srcs and coords per exp
                   chunk['nrec'], chunk['recy'], chunk['recx'], # Num recs and coords per exp
                   chunk['dat'] , chunk['wav'],  img,           # Data, wavelets and output images
                   self.__nthrds, self.__verb)                  # Threading and verbosity

    return img

  def image_data(self,dat,dt,minf,maxf,vel,jf=1,nhx=0,nhy=0,sym=True,nrmax=3,eps=0.0,dtmax=5e-05,wav=None,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False):
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

  def get_off_axis(self):
    """ Returns the x subsurface offset extension axis """
    if(self.__rnhx is None):
      raise Exception("Cannot return x subsurface offset axis without running extended imaging")
    return self.__rnhx, self.__ohx, self.__dhx

  def fft1(self,sig,dt,minf,maxf,jf=1,save=True):
    """
    Computes the FFT along the fast axis. Input
    array can be N-dimensional

    Parameters:
      sig  - the input time-domain signal (time is fast axis)
      dt   - temporal sampling of input data
      minf - the minimum frequency for windowing the spectrum [Hz]
      maxf - the maximum frequency for windowing the spectrum
      save - save the frequency axes [True]

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
    sigfft  = np.fft.fft(sigp)[...,begw:endw]
    # Subsample frequencies
    sigfftd = sigfft[::jf]
    # Set frequency axis
    if(save):
      # Frequency axes
      self.__nwc = sigfftd.shape[0]
      self.__dwc = jf*dw
      self.__nwo = nw; self.__ow = minf; self.__dw = dw
      return sigfftd.astype('complex64')
    else:
      return nw,minf,dw,sigfftd.astype('complex64')

  def get_freq_axis(self):
    """ Returns the data frequency axis """
    return self.__nwo,self.__ow,self.__dw

  def ifft1(self,sig,nw,ow,dw,n1,it0=0):
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

  def splitnum(self,num,div):
    """ Splits a number into nearly even parts """
    splits = []
    igr,rem = divmod(num,div)
    for i in range(div):
      splits.append(igr)
    for i in range(rem):
      splits[i] += 1

    return splits

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
    # Assume ny = 1
    imgin = img[0,:,:,0,:]
    amin = -amax; avals = np.linspace(amin,amax,na)
    # Compute angle axis
    self.__na = na; self.__da = avals[1] - avals[0]; self.__oa = avals[0]
    return off2ang(imgin,self.__ohx,self.__dhx,self.__dz,na=na,amax=amax,nta=601,ota=-3,dta=0.01,
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

