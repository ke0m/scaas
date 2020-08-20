"""
Imaging/modeling based on source and receiver coordinates
This code is designed to be distributed across nodes in a cluster

@author: Joseph Jennings
@version: 2020.08.17
"""
import numpy as np
from oway.ssr3 import ssr3, interp_slow
from server.utils import splitnum
from scaas.off2ang import off2angssk,off2angkzx
from genutils.ptyprint import progressbar
import matplotlib.pyplot as plt

def default_coord(nx,dx,ny,dy,nz,dz,
                  nsx,dsx,nsy,dsy,osx=0.0,osy=0.0,
                  nrx=None,drx=1.0,orx=0.0,nry=None,dry=1.0,ory=0.0):
  """
  Makes coordinates based on a default coordinate geometry
  (split spread with regular source and receiver spacing)

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

  Returns srcx, srcy, nrec, recx, recy
  """
  # Number of experiments
  nexp = nsx*nsy

  # Source coordinates
  srcx = np.zeros(nexp,dtype='float32')
  srcy = np.zeros(nexp,dtype='float32')
  iexp = 0
  for isy in range(nsy):
    sy = int(osy + isy*dsy)
    for isx in range(nsx):
      sx = int(osx + isx*dsx)
      srcx[iexp] = sx*dx
      srcy[iexp] = sy*dy
      iexp += 1

  # Receiver coordinates
  if(nrx is None): nrx = nx
  if(nry is None): nry = ny
  inrec = int(nrx*nry)
  irecx = np.zeros(inrec,dtype='float32')
  irecy = np.zeros(inrec,dtype='float32')
  nrec  = np.zeros(nexp,dtype='int32')
  nrec[:] = inrec

  # Compute the receivers for one shot
  itr = 0
  for iry in range(nry):
    ry = int(ory + iry*dry)
    for irx in range(nrx):
      rx = int(orx + irx*drx)
      irecx[itr] = rx*dx
      irecy[itr] = ry*dy
      itr += 1

  # Repeat for all shots
  recx = np.tile(irecx,nexp)
  recy = np.tile(irecy,nexp)

  return srcx,srcy,nrec,recx,recy

class coordgeomchunk:
  """
  Functions for modeling and imaging with a
  field data (coordinate) geometry
  """
  def __init__(self,nx,dx,ny,dy,nz,dz,
               nrec,srcx=None,srcy=None,recx=None,recy=None,
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
      srcx  - x coordinates of source locations [number of shots]
      srcy  - y coordinates of source locations [number of shots]
      recx  - x coordinates of receiver locations [number of traces]
      recy  - y coordinates of receiver locations [number of traces]

    Returns:
      a coordinate geom object
    """
    # Spatial axes
    self.__nx = nx; self.__ox = ox; self.__dx = dx
    self.__ny = ny; self.__oy = oy; self.__dy = dy
    self.__nz = nz; self.__oz = oz; self.__dz = dz
    ## Source gometry
    # Check if either is none
    if(srcx is None and srcy is None):
      raise Exception("Must provide either srcx or srcy coordinates")
    if(srcx is None):
      srcx = np.zeros(len(srcy),dtype='int')
    if(srcy is None):
      srcy = np.zeros(len(srcx),dtype='int')
    # Make sure coordinates are within the model
    if(np.any(srcx >= ox+(nx)*dx) or np.any(srcy >= oy+(ny)*dy)):
      print("Warning: Some source coordinates are greater than model size")
    if(np.any(srcx < ox) or np.any(srcy <  oy)):
      print("Warning: Some source coordinates are less than model size")
    if(len(srcx) != len(srcy)):
      raise Exception("Length of srcx must equal srcy")
    self.__srcx = srcx.astype('float32'); self.__srcy = srcy.astype('float32')
    # Total number of sources
    self.__nexp = len(srcx)
    # Assume one source per shot
    self.__nsrc = np.ones(self.__nexp,dtype='int32')
    ## Receiver geometry
    # Check if either is none
    if(recx is None and recy is None):
      raise Exception("Must provide either recx or recy coordinates")
    if(recx is None):
      recx = np.zeros(len(recy),dtype='int')
    if(recy is None):
      recy = np.zeros(len(recx),dtype='int')
    # Make sure coordinates are within the model
    if(np.any(recx >= ox + nx*dx) or np.any(recy >= oy + ny*dy)):
      print("Warning: Some receiver coordinates are greater than model size")
    if(np.any(recx < ox) or np.any(recy <  oy)):
      print("Warning: Some receiver coordinates are less than model size")
    if(len(recx) != len(recy)):
      raise Exception("Each trace must have same number of x and y coordinates")
    self.__recx = recx.astype('float32'); self.__recy = recy.astype('float32')
    # Number of receivers per shot
    if(nrec.dtype != 'int' and  nrec.dtype != 'int32'):
      raise Exception("nrec (number of receivers) must be integer type array")
    self.__nrec = nrec
    # Number of traces
    self.__ntr = len(recx)

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

  def model_data(self,wav,owc,dwc,vel,ref,
                 nrmax=3,eps=0.,dtmax=5e-05,ntx=0,nty=0,px=0,py=0,
                 nthrds=1,sverb=True,wverb=True):
    """
    3D modeling of single scattered (Born) data with the one-way
    wave equation (single square root (SSR), split-step Fourier method).

    Parameters:
      wav    - the input wavelet (source time function) [nt]
      vel    - input velocity model [nz,ny,nx]
      ref    - input reflectivity model [nz,ny,nx]
      nrmax  - maximum number of reference velocities [3]
      eps    - stability parameter [0.]
      dtmax  - maximum time error [5e-05]
      ntx    - size of taper in x direction (samples) [0]
      nty    - size of taper in y direction (samples) [0]
      px     - amount of padding in x direction (samples)
      py     - amount of padding in y direction (samples)
      nthrds - number of OpenMP threads to use for frequency parallelization [1]
      sverb  - verbosity flag for shot progress bar [True]
      wverb  - verbosity flag for frequency progress bar [False]

    Returns:
      the data at the surface [nw,nry,nrx]
    """
    # Get dimensions
    nwc = wav.shape[0]

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz ,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz ,     # Spatial Samplings
               nwc,owc,dwc,eps,                     # Frequency axis
               ntx,nty,px,py,                       # Taper and padding
               dtmax,nrmax,nthrds)                  # Reference velocities and threads

    # Compute slowness and reference slownesses
    slo = 1/vel
    ssf.set_slows(slo)

    # Allocate output data (surface wavefield) and receiver data
    datw  = np.zeros([nwc,self.__ny,self.__nx],dtype='complex64')
    recw  = np.zeros([self.__ntr,nwc],dtype='complex64')

    # Allocate the source for one shot
    sou = np.zeros([nwc,self.__ny,self.__nx],dtype='complex64')

    # Loop over sources
    ntr = 0
    for iexp in progressbar(range(self.__nexp),"nexp:",verb=sverb):
      # Get the source coordinates
      sy = self.__srcy[iexp]; sx = self.__srcx[iexp]
      isy = int((sy-self.__oy)/self.__dy+0.5); isx = int((sx-self.__ox)/self.__dx+0.5)
      # Create the source for this shot
      sou[:] = 0.0
      sou[:,isy,isx]  = wav[:]
      # Downward continuation
      datw[:] = 0.0
      ssf.modallw(ref,sou,datw,wverb)
      # Restrict to receiver locations
      datwt = np.ascontiguousarray(np.transpose(datw,(1,2,0)))  # [nwc,ny,nx] -> [ny,nx,nwc]
      ssf.restrict_data(self.__nrec[iexp],self.__recy[ntr:],self.__recx[ntr:],self.__oy,self.__ox,datwt,recw[ntr:,:])
      # Increase number of traces
      ntr += self.__nrec[iexp]

    return recw

  def image_data(self,rec,owc,dwc,vel,
                 nhx=0,nhy=0,sym=True,eps=0,nrmax=3,dtmax=5e-05,wav=None,
                 ntx=0,nty=0,px=0,py=0,nthrds=1,sverb=True,wverb=False):
    """
    3D migration of shot profile data via the one-way wave equation (single-square
    root split-step fourier method). Input data are assumed to follow
    the default geometry (sources and receivers on a regular grid)

    Parameters:
      rec    - flattened input shot profile data
      vel    - input migration velocity model [nz,ny,nx]
      jf     - frequency decimation factor
      nhx    - number of subsurface offsets in x to compute [0]
      nhy    - number of subsurface offsets in y to compute [0]
      sym    - symmetrize the subsurface offsets [True]
      nrmax  - maximum number of reference velocities [3]
      dtmax  - maximum time error [5e-05]
      wav    - input wavelet [None,assumes an impulse at zero lag]
      ntx    - size of taper in x direction [0]
      nty    - size of taper in y direction [0]
      px     - amount of padding in x direction (samples) [0]
      py     - amount of padding in y direction (samples) [0]
      nthrds - number of OpenMP threads for parallelizing over frequency [1]
      sverb  - verbosity flag for shot progress bar [True]
      wverb  - verbosity flag for frequency progress bar [False]
      nchnks - number of chunks to distribute over cluster
      client - dask client for distributing work over a cluster

    Returns:
      an image created from the data [nhy,nhx,nz,ny,nx]
    """
    # Make sure data are same size as coordinates
    if(rec.shape[0] != self.__ntr):
      raise Exception("Data must have same number of traces passed to constructor")

    nwc = wav.shape[0]
    if(rec.shape[-1] != nwc):
      raise Exception("Data and wavelet must have same frequency axis")

    # Allocate source and data for one shot
    datw = np.zeros([self.__ny,self.__nx,nwc],dtype='complex64')
    sou  = np.zeros([nwc,self.__ny,self.__nx],dtype='complex64')

    # Single square root object
    ssf = ssr3(self.__nx ,self.__ny,self.__nz ,     # Spatial Sizes
               self.__dx ,self.__dy,self.__dz ,     # Spatial Samplings
               nwc,owc,dwc,eps,                     # Frequency axis
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
      sy = self.__srcy[iexp]; sx = self.__srcx[iexp]
      isy = int((sy-self.__oy)/self.__dy+0.5); isx = int((sx-self.__ox)/self.__dx+0.5)
      # Create the source wavefield for this shot
      sou[:] = 0.0
      sou[:,isy,isx]  = wav[:]
      # Inject the data for this shot
      datw[:] = 0.0
      ssf.inject_data(self.__nrec[iexp],self.__recy[ntr:],self.__recx[ntr:],self.__oy,self.__ox,rec[ntr:,:],datw)
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

  def plot_acq(self,mod=None,show=True,**kwargs):
    """ Plots the acquisition on the velocity model for a specified shot """
    pass

