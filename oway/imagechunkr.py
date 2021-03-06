"""
Chunks imaging inputs and parameters
for distribution across multiple machines

@author: Joseph Jennings
@version: 2020.08.18
"""
import numpy as np
from oway.utils import fft1, interp_vel
from server.utils import splitnum

class imagechunkr:

  def __init__(self,nchnks,
               nx,dx,ny,dy,nz,dz,
               vel,dat,dt,minf,maxf,
               nrec,srcx=None,srcy=None,recx=None,recy=None,
               ox=0.0,oy=0.0,oz=0.0,dvx=None,ovx=0.0,dvy=None,ovy=0.0,
               wav=None,jf=1,verb=True):
    """
    Creates a generator from inputs necessary
    for imaging data

    Parameters:
      nchnks - length of generator (number of chunks to yield)
      nx     - number of x samples of the output image
      dx     - x sampling of the reflectivity
      ny     - number of y samples of the output image
      dy     - y sampling of the reflectivity
      nz     - number of z samples of the output image
      dz     - z sampling of the velocity model
      vel    - input velocity model
      dat    - input data
      dt     - temporal sampling of data
      minf   - minimum frequency to use for imaging
      maxf   - maximum frequency to use for imaging
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
      wav    - input wavelet [default is delta function at zero lag]
      jf     - subsampling of frequency axis [1]
      verb   - prints chunking and frequency information [True]
    """
    # Number of chunks to create (length of generator)
    self.__nchnks = nchnks

    # Reflectivity and dimensions
    self.__nz,self.__ny,self.__nx = nz,ny,nx
    self.__oz,self.__oy,self.__ox = oz,oy,ox
    self.__dz,self.__dy,self.__dx = dz,dy,dx

    # Check source coordinates
    self.__srcx,self.__srcy = srcx,srcy
    if(self.__srcx is None and self.__srcy is None):
      raise Exception("Must provide either srcx or srcy coordinates")
    if(self.__srcx is None):
      self.__srcx = np.zeros(len(self.__srcy),dtype='int')
    if(self.__srcy is None):
      self.__srcy = np.zeros(len(self.__srcx),dtype='int')
    if(len(self.__srcx) != len(self.__srcy)):
      raise Exception("Number of srcx coordinates must == number of srcy coordinates")

    # Check receiver coordinates
    self.__recx,self.__recy,self.__nrec = recx,recy,nrec
    if(self.__recx is None and self.__recy is None):
      raise Exception("Must provide either recx or recy coordinates")
    if(self.__recx is None):
      self.__recx = np.zeros(len(self.__recy),dtype='int')
    if(recy is None):
      self.__recy = np.zeros(len(self.__recx),dtype='int')
    if(len(self.__recx) != len(self.__recy)):
      raise Exception("Number of recx coordinates must == number of recy coordinates")

    # Get number of experiments
    self.__nexp = len(nrec)
    if(len(self.__nexp) != len(self.__srcx)):
      raise Exception("Number of nrecs must equal number of srcx/y coordinates")

    # Create frequency domain source
    self.__nt,self.__dt = dat.shape[-1],dt
    if(wav is None):
      wav    = np.zeros(self.__nt,dtype='float32')
      wav[0] = 1.0
    self.__nwo,self.__ow,self.__dw,wfft = fft1(wav,dt,minf=minf,maxf=maxf)
    self.__wfftd = wfft[::jf]
    self.__nwc = self.__wfftd.shape[0] # Get the number of frequencies for imaging
    self.__dwc = self.__dw*jf

    # Create frequency domain data
    _,_,_,dfft = fft1(dat,dt,minf=minf,maxf=maxf)
    self.__dfftd = dfft[:,::jf]

    if(verb): print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))

    # Interpolate the velocity if needed
    if(vel.shape != (self.__nz,self.__ny,self.__nx)):
      if(dvx is None and dvy is None):
        raise Exception("If vel shape != output image shape, must provide dvx or dvy")
      if(dvy is None and self.__ny == 1): dvy = 1.0
      if(dvx is None and self.__nx == 1): dvx = 1.0

      self.__vel = interp_vel(self.__nz,
                              self.__ny,self.__oy,self.__dy,
                              self.__nx,self.__ox,self.__dx,
                              vel,dvx,dvy,ovx,ovy)

    else:
      self.__vel = vel

    # Default imaging parameters
    self.__nrmax, self.__dtmax, self.__eps  = 3, 5e-05, 0.0
    self.__ntx, self.__nty = 0, 0
    self.__px,  self.__py  = 0, 0
    # Extended imaging parameters
    self.__nhx, self.__ohx  = 0, 0.0
    self.__nhy, self.__ohy  = 0, 0.0
    self.__sym  = True
    self.__rnhx, self.__rnhy = None, None
    # Verbosity and threading
    self.__nthrds = 1
    self.__wverb,self.__everb  = False, False

  def set_image_pars(self,nrmax=3,dtmax=5e-05,eps=0.0,
                     ntx=0,nty=0,px=0,py=0,
                     nhx=0,nhy=0,sym=True,
                     nthrds=1,wverb=False,sverb=False):
    """
    Overrides default parameters set in the constructor for the modeling parameters

    Parameters:
      nrmax - maximum number of reference velocities [3]
      dtmax - maximum time error [5e-5]
      eps    - stability parameter
      ntx    - size of taper in x direction (samples) [0]
      nty    - size of taper in y direction (samples) [0]
      px     - amount of padding in x direction (samples)
      py     - amount of padding in y direction (samples)
      nthrds - number of OpenMP threads to use for frequency parallelization [1]
      sverb  - verbosity flag for shot progress bar [False]
      wverb  - verbosity flag for frequency progress bar [False]
    """
    self.__nrmax, self.__dtmax, self.__eps  = nrmax, dtmax, eps
    self.__ntx, self.__nty = ntx, nty
    self.__px,  self.__py  = px, py
    # Extended imaging parameters
    self.__nhx, self.__nhy, self.__sym = nhx, nhy, sym
    # Verbosity and threads
    self.__nthrds = nthrds
    self.__wverb,self.__sverb = wverb, sverb

  def get_img_shape(self):
    """ Returns the shape of the output image """
    if(self.__nhx == 0 and self.__nhy == 0):
      return [self.__nz,self.__ny,self.__nx]
    else:
      if(self.__sym):
        self.__rnhx, self.__rnhy = 2*self.__nhx+1, 2*self.__nhy+1
      else:
        self.__rnhx, self.__rnhy = self.__nhx+1, self.__nhy+1
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

    To create the generator, use gen = iter(modelchunkr(args))
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
      rychnk  = self.__recy [begr:endr]
      rxchnk  = self.__recx [begr:endr]
      datchnk = self.__dfftd[begr:endr,:]
      # Update positions
      begs,begr = ends, endr
      ## Constructor arguments
      cdict = {}
      # Parameters for constructor
      cdict['nx']   = self.__nx;  cdict['ox']   = self.__ox;  cdict['dx'] = self.__dx
      cdict['ny']   = self.__ny;  cdict['oy']   = self.__oy;  cdict['dy'] = self.__dy
      cdict['nz']   = self.__nz;  cdict['oz']   = self.__oz;  cdict['dz'] = self.__dz
      cdict['srcy'] = sychnk;     cdict['srcx'] = sxchnk
      cdict['recy'] = rychnk;     cdict['recx'] = rxchnk
      cdict['nrec'] = nreccnk
      ## Imaging arguments
      idict = {}
      # Parameters for imaging
      idict['nrmax']  = self.__nrmax;  idict['dtmax'] = self.__dtmax; idict['eps'] = self.__eps
      idict['ntx']    = self.__ntx;    idict['nty']   = self.__nty;
      idict['px']     = self.__px;     idict['py']    = self.__py;
      idict['nthrds'] = self.__nthrds
      idict['sverb']  = self.__sverb;  idict['wverb'] = self.__wverb
      # Extended imaging parameters
      idict['nhx']    = self.__nhx;    idict['nhy']   = self.__nhy;   idict['sym'] = self.__sym
      # Frequency domain axis
      idict['dwc']  = self.__dwc;      idict['owc']   = self.__ow
      # Imaging inputs
      idict['wav']  = self.__wfftd;    idict['vel']   = self.__vel;   idict['rec']  = datchnk
      yield [cdict,idict,ichnk]
      ichnk += 1

