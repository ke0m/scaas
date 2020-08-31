"""
Chunks modeling and migration (hessian) inputs and parameters
for distribution across multiple machines

@author: Joseph Jennings
@version: 2020.08.23
"""
import numpy as np
from oway.utils import fft1, ifft1, interp_vel
from server.utils import splitnum

class hessnchunkr:

  def __init__(self,nchnks,
               dx,dy,dz,
               ref,velmod,velmig,modwav,dt,t0,minf,maxf,
               nrec,srcx=None,srcy=None,recx=None,recy=None,
               ox=0.0,oy=0.0,oz=0.0,dvx=None,ovx=0.0,dvy=None,ovy=0.0,
               migwav=None,jf=1):
    """
    Creates a generator from inputs necessary
    for computing the Hessian

    Parameters:
      nchnks - length of generator (number of chunks to yield)
      dx     - x sampling of the input reflectivity
      dy     - y sampling of the input reflectivity
      dz     - z sampling of the input reflectivity
      ref    - input reflectivity model
      velmod - modeling velocity
      velmig - migration velocity
      wav    - the input wavelet for modeling
      dt     - temporal sampling of wavelet
      t0     - wavelet time zero(seconds)
      minf   - minimum frequency to use for modeling
      maxf   - maximum frequency to use for modeling
      nrec   - number of receivers per shot [number of shots]
      srcx   - x coordinates of source locations [number of shots]
      srcy   - y coordinates of source locations [number of shots]
      recx   - x coordinates of receiver locations [number of traces]
      recy   - y coordinates of receiver locations [number of traces]
      ox     - reflectivity origin on x axis [0.0]
      oy     - reflectivity origin on y axis [0.0]
      oz     - reflectivity origin on z axis [0.0]
      dvx    - x sampling of velocity model
      ovx    - x origin of velocity model
      dvy    - y sampling of velocity model
      ovy    - y origin of velocity model
      migwav - wavelet for migration
      jf     - subsampling of frequency axis
    """
    # Number of chunks to create (length of generator)
    self.__nchnks = nchnks

    # Reflectivity and dimensions
    self.__ref = ref
    [self.__nz,self.__ny,self.__nx] = ref.shape
    self.__oz = oz; self.__oy = oy; self.__ox = ox
    self.__dz = dz; self.__dy = dy; self.__dx = dx

    # Check source coordinates
    self.__srcx = srcx; self.__srcy = srcy
    if(self.__srcx is None and self.__srcy is None):
      raise Exception("Must provide either srcx or srcy coordinates")
    if(self.__srcx is None):
      self.__srcx = np.zeros(len(self.__srcy),dtype='int')
    if(self.__srcy is None):
      self.__srcy = np.zeros(len(self.__srcx),dtype='int')
    if(len(self.__srcx) != len(self.__srcy)):
      raise Exception("srcx and srcy are not same length")

    # Check receiver coordinates
    self.__recx = recx; self.__recy = recy; self.__nrec = nrec
    if(self.__recx is None and self.__recy is None):
      raise Exception("Must provide either recx or recy coordinates")
    if(self.__recx is None):
      self.__recx = np.zeros(len(self.__recy),dtype='int')
    if(recy is None):
      self.__recy = np.zeros(len(self.__recx),dtype='int')

    # Get number of experiments
    self.__nexp = len(nrec)

    # Create the input frequency modeling domain source and get original frequency axis
    self.__nt = modwav.shape[0]; self.__dt = dt; self.__t0 = t0
    self.__nwo,self.__ow,self.__dw,wfft = fft1(modwav,dt,minf=minf,maxf=maxf)
    self.__modwfftd = wfft[::jf]
    self.__nwc = self.__modwfftd.shape[0] # Get the number of frequencies to compute
    self.__dwc = jf*self.__dw

    # Source for migration
    if(migwav is None):
      migwav = np.zeros(self.__nt,dtype='float32')
      migwav[0] = 1.0
    _,_,_,wfft = fft1(migwav,dt,minf=minf,maxf=maxf)
    self.__migwfftd = wfft[::jf]

    # Interpolate the velocity if needed
    if(velmod.shape != ref.shape or velmig.shape != ref.shape):
      if(dvx is None or dvy is None):
        raise Exception("If vel shape != ref shape, must provide dvx or dvy")

      self.__velmod = interp_vel(self.__nz,
                                 self.__ny,self.__oy,self.__dy,
                                 self.__nx,self.__ox,self.__dx,
                                 velmod,dvx,dvy,ovx,ovy)

      self.__velmig = interp_vel(self.__nz,
                                 self.__ny,self.__oy,self.__dy,
                                 self.__nx,self.__ox,self.__dx,
                                 velmig,dvx,dvy,ovx,ovy)

    else:
      self.__velmod = velmod
      self.__velmig = velmig

    # Default modeling/imaging parameters
    self.__nrmax  = 3; self.__dtmax = 5e-05; self.__eps = 0.0
    self.__ntx    = 0; self.__nty   = 0
    self.__mpx    = 0; self.__mpy   = 0
    self.__ipx    = 0; self.__ipy   = 0
    # Extended imaging parameters
    self.__nhx = 0; self.__ohx = 0.0
    self.__nhy = 0; self.__ohy = 0.0
    self.__sym = True
    # Verbosity and threading
    self.__nthrds = 1
    self.__wverb  = False; self.__sverb = False

  def set_hessn_pars(self,nrmax=3,dtmax=5e-05,eps=0.0,
                     ntx=0,nty=0,mpx=0,mpy=0,ipx=0,ipy=0,
                     nhx=0,nhy=0,sym=True,
                     nthrds=1,wverb=False,sverb=False):
    """
    Overrides default parameters set in the constructor for the Hessian parameters

    Parameters:
      nrmax   - maximum number of reference velocities [3]
      dtmax   - maximum time error [5e-5]
      eps     - stability parameter
      ntx     - size of taper in x direction (samples) [0]
      nty     - size of taper in y direction (samples) [0]
      mpx     - amount of padding in x direction for modeling (samples)
      mpy     - amount of padding in y direction for modeling (samples)
      ipx     - amount of padding in x direction for imaging (samples)
      ipy     - amount of padding in y direction for imaging  (samples)
      nhx     - number of x subsurface offsets [0]
      nhy     - number of y subsurface offsets [0]
      sym     - whether to compute a symmetric cross-correlation
      nthrds  - number of OpenMP threads to use for frequency parallelization [1]
      sverb   - verbosity flag for shot progress bar [False]
      wverb   - verbosity flag for frequency progress bar [False]
    """
    self.__nrmax  = nrmax; self.__dtmax = dtmax; self.__eps = eps
    self.__ntx    = ntx; self.__nty   = nty
    self.__mpx    = mpx; self.__mpy   = mpy
    self.__ipx    = ipx; self.__ipy   = ipy
    # Extended imaging parameters
    self.__nhx = nhx; self.__nhy = nhy; self.__sym = sym
    # Verbosity and threads
    self.__nthrds = nthrds
    self.__wverb  = wverb; self.__sverb = sverb

  def get_img_shape(self):
    """ Returns the shape of the output image """
    if(self.__nhx == 0 and self.__nhy == 0):
      return [self.__nz,self.__ny,self.__nx]
    else:
      if(self.__sym):
        self.__rnhx = 2*self.__nhx+1; self.__rnhy = 2*self.__nhy+1
      else:
        self.__rnhx = self.__nhx+1; self.__rnhy = self.__nhy+1
      return [self.__rnhy,self.__rnhx,self.__nz,self.__ny,self.__nx]

  def get_offx_axis(self):
    """ Returns the x subsurface offset axes """
    if(self.__rnhx is None):
      raise Exception("Cannot return x subsurface offset axis without running extended imaging")
    if(self.__sym): ohx = -self.__nhx*self.__dx
    else: ohx = 0.0

    return self.__rnhx, ohx, self.__dx

  def __iter__(self):
    """
    Defines the iterator for creating chunks

    To create the generator, use gen = iter(hessnchunkr(args))
    """
    # Number of shots per chunk
    expchnks = splitnum(self.__nexp,self.__nchnks)

    k = 0
    begs = 0; ends = 0; begr = 0; endr = 0
    ichnk = 0
    while ichnk < len(expchnks):
      # Get data and sources for each chunk
      nreccnk = np.zeros(expchnks[ichnk],dtype='int32')
      for iexp in range(expchnks[ichnk]):
        nreccnk[iexp] = self.__nrec[k]
        endr += self.__nrec[k]
        ends += 1
        k += 1
      # Chunked source data
      sychnk  = self.__srcy[begs:ends]
      sxchnk  = self.__srcx[begs:ends]
      # Chunked receiver data
      rychnk  = self.__recy[begr:endr]
      rxchnk  = self.__recx[begr:endr]
      # Update positions
      begs = ends; begr = endr
      ## Constructor arguments
      cdict = {}
      # Parameters for constructor
      cdict['nx']   = self.__nx;  cdict['ox']   = self.__ox;  cdict['dx']   = self.__dx
      cdict['ny']   = self.__ny;  cdict['oy']   = self.__oy;  cdict['dy']   = self.__dy
      cdict['nz']   = self.__nz;  cdict['oz']   = self.__oz;  cdict['dz']   = self.__dz
      cdict['srcy'] = sychnk;     cdict['srcx'] = sxchnk
      cdict['recy'] = rychnk;     cdict['recx'] = rxchnk
      cdict['nrec'] = nreccnk
      ## Modeling arguments
      mdict = {}
      # Parameters for modeling
      mdict['nrmax']  = self.__nrmax;  mdict['dtmax'] = self.__dtmax; mdict['eps']  = self.__eps
      mdict['ntx']    = self.__ntx;    mdict['nty']   = self.__nty;
      mdict['px']     = self.__mpx;    mdict['py']    = self.__mpy;
      mdict['nthrds'] = self.__nthrds
      mdict['sverb']  = self.__sverb;  mdict['wverb'] = self.__wverb
      # Frequency domain axis and delay
      mdict['dwc']  = self.__dwc;      mdict['owc']   = self.__ow;     mdict['t0']   = self.__t0
      # Modeling inputs
      mdict['wav']  = self.__modwfftd; mdict['vel']   = self.__velmod; mdict['ref']  = self.__ref
      ## Imaging arguments
      idict = {}
      # Parameters for imaging
      idict['nrmax']  = self.__nrmax;  idict['dtmax'] = self.__dtmax; idict['eps']  = self.__eps
      idict['ntx']    = self.__ntx;    idict['nty']   = self.__nty
      idict['px']     = self.__ipx;    idict['py']    = self.__ipy
      idict['nthrds'] = self.__nthrds
      idict['sverb']  = self.__sverb;  idict['wverb'] = self.__wverb
      # Extended imaging parameters
      idict['nhx'] = self.__nhx;       idict['nhy'] = self.__nhy;     idict['sym'] = self.__sym
      # Frequency domain axis
      idict['dwc']  = self.__dwc;      idict['owc']   = self.__ow
      # Imaging inputs
      idict['wav'] = self.__migwfftd;  idict['vel'] = self.__velmig
      yield [cdict,mdict,idict,ichnk]
      ichnk += 1

