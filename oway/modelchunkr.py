"""
Chunks modeling inputs and parameters
for distribution across multiple machines

@author: Joseph Jennings
@version: 2020.08.31
"""
import numpy as np
from oway.utils import fft1, ifft1, interp_vel, make_sht_cube
from server.utils import splitnum

class modelchunkr:

  def __init__(self,nchnks,
               dx,dy,dz,
               ref,vel,wav,dt,minf,maxf,
               nrec,srcx=None,srcy=None,recx=None,recy=None,
               ox=0.0,oy=0.0,oz=0.0,dvx=None,ovx=0.0,dvy=None,ovy=0.0,
               jf=1,t0=0,verb=True):
    """
    Creates a generator from inputs necessary
    for modeling data

    Parameters:
      nchnks - length of generator (number of chunks to yield)
      dx     - x sampling of the input reflectivity
      dy     - y sampling of the input reflectivity
      dz     - z sampling of the input reflectivity
      ref    - input reflectivity model
      vel    - input velocity model
      wav    - the input wavelet for modeling
      dt     - temporal sampling of wavelet
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
      jf     - subsampling of frequency axis
      t0     - time zero of the input wavelet [0]
      verb   - verbosity flag [False]
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

    # Create the input frequency domain source and get original frequency axis
    self.__nt = wav.shape[0]; self.__dt = dt; self.__t0 = t0
    self.__nwo,self.__ow,self.__dw,wfft = fft1(wav,dt,minf=minf,maxf=maxf)
    self.__wfftd = wfft[::jf]
    self.__nwc = self.__wfftd.shape[0] # Get the number of frequencies to compute
    self.__dwc = jf*self.__dw

    if(verb): print("Frequency axis: nw=%d ow=%f dw=%f"%(self.__nwc,self.__ow,self.__dwc))

    # Interpolate the velocity if needed
    if(vel.shape != ref.shape):
      if(dvx is None and dvy is None):
        raise Exception("If vel shape != ref shape, must provide dvx or dvy")
      if(dvy is None and self.__ny == 1): dvy = 1.0
      if(dvx is None and self.__nx == 1): dvx = 1.0

      self.__vel = interp_vel(self.__nz,
                              self.__ny,self.__oy,self.__dy,
                              self.__nx,self.__ox,self.__dx,
                              vel,dvx,dvy,ovx,ovy)
    else:
      self.__vel = vel

    # Default modeling parameters
    self.__nrmax  = 3; self.__dtmax = 5e-05; self.__eps = 0.0
    self.__ntx    = 0; self.__nty   = 0
    self.__px     = 0; self.__py    = 0
    self.__nthrds = 1
    self.__wverb  = False; self.__sverb = False

  def set_model_pars(self,nrmax=3,dtmax=5e-05,eps=0.0,
                     ntx=0,nty=0,px=0,py=0,
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
    self.__nrmax  = nrmax; self.__dtmax = dtmax; self.__eps = eps
    self.__ntx    = ntx; self.__nty   = nty
    self.__px     = px; self.__py     = py
    self.__nthrds = nthrds
    self.__wverb  = wverb; self.__sverb = sverb

  def get_freq_axis(self):
    """
    Returns the frequency axis
    """
    return self.__nwo,self.__ow,self.__dw

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
      rychnk  = self.__recy[begr:endr]
      rxchnk  = self.__recx[begr:endr]
      # Update positions
      begs = ends; begr = endr
      ## Constructor arguments
      cdict = {}
      # Parameters for constructor
      cdict['nx']   = self.__nx;  cdict['ox']   = self.__ox;  cdict['dx'] = self.__dx
      cdict['ny']   = self.__ny;  cdict['oy']   = self.__oy;  cdict['dy'] = self.__dy
      cdict['nz']   = self.__nz;  cdict['oz']   = self.__oz;  cdict['dz'] = self.__dz
      cdict['srcy'] = sychnk;     cdict['srcx'] = sxchnk
      cdict['recy'] = rychnk;     cdict['recx'] = rxchnk
      cdict['nrec'] = nreccnk
      ## Modeling arguments
      mdict = {}
      # Parameters for modeling
      mdict['nrmax']  = self.__nrmax;  mdict['dtmax'] = self.__dtmax; mdict['eps'] = self.__eps
      mdict['ntx']    = self.__ntx;    mdict['nty']   = self.__nty;
      mdict['px']     = self.__px;     mdict['py']    = self.__py;
      mdict['nthrds'] = self.__nthrds
      mdict['sverb']  = self.__sverb;  mdict['wverb'] = self.__wverb
      # Frequency domain axis
      mdict['dwc']  = self.__dwc;      mdict['owc']   = self.__ow;    mdict['t0']  = self.__t0
      # Modeling inputs
      mdict['wav']  = self.__wfftd;    mdict['vel']   = self.__vel;   mdict['ref'] = self.__ref
      yield [cdict,mdict,ichnk]
      ichnk += 1

  def reconstruct_data(self,chunks,dly,time=True,reg=False):
    """
    Reorder the shots after collecting. Also computes
    inverse FFT and regularizes if desired

    Parameters:
      chunks - the collected data chunks (dictionary)
               (assumes has keys 'result' and 'ep')
      dly    - the time delay of the wavelet (seconds)
      time   - inverse FFT to convert to time [True]
      reg    - make a regularized cube of the data [False]
    """
    # Sort the data based on the chunk IDs
    idx  = np.argsort(chunks['cid'])
    dats = np.concatenate(np.asarray(chunks['result'])[idx],axis=0)
    if(time):
      # Inverse FFT
      it0 = int(dly/self.__dt)
      dato = ifft1(dats,self.__nwo,self.__ow,self.__dw,self.__nt,it0)
    else:
      dato = dats
    if(reg):
      # Make data a regular cube if desired
      datg = make_sht_cube(self.__nrec,dato)
    else:
      datg = dato

    return datg

