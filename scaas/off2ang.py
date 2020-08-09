"""
Functions for converting subsurface offset to angle gathers
@author: Joseph Jennings
@version: 2020.08.09
"""
import numpy as np
from scaas.adcig import convert2ang
from scaas.adcigkzkx import convert2angkzkykx
from utils.ptyprint import printprogress, progressbar

def off2angssk(off,oh,dh,dz,oz=0.0,na=281,amax=70,oa=None,da=None,nta=601,ota=-3,dta=0.01,
              nthrds=4,transp=False,oro=None,dro=None,verb=False):
  """
  Convert subsurface offset gathers to opening angle gathers

  Parameters
    off    - Subsurface offset gathers of shape [nro,nh,nz,nx] (nro is optional)
    oh     - Origin of subsurface offset axis
    dh     - Sampling of subsurface offset axis
    dz     - Sampling in depth
    oz     - Depth origin [0.0]
    na     - Number of angles [281]
    amax   - Maximum angle [70 degrees]
    oa     - Origin of angle axis (not needed if na and amax specified)
    da     - Spacing on angle axis (not needed if na and amax specified)
    nta    - Number of slant-stacks (tangents) to be computed [601]
    ota    - Origin of slant-stack (tangent) axis [-3]
    dta    - Sampling of slant-stack tangent axis [0.01]
    nthrds - Number of OpenMP threads to parallelize over image point axis [4]
    transp - Transpose the output to have shape [nro,na,nz,nx]
    oro    - Origin of rho axis (triggers another type of verbosity) [None]
    dro    - Sampling of rho axis (triggers another type of verbosity) [None]
    verb   - Verbosity flag [False]

  Returns the converted angle gathers of shape [nro,nx,na,nz]
  """
  # Handle the case if no rho axis
  if(len(off.shape) == 3):
    offr = np.expand_dims(off,axis=0)
  else:
    offr = off
  # Transpose the data to have shape [nr,nx,nh,nz]
  offt = np.ascontiguousarray(np.transpose(offr,(0,3,1,2)))
  # Get shape of data
  nro = offt.shape[0]; nx = offt.shape[1]; nh = offt.shape[2]; nz = offt.shape[3]
  # Compute the angle axis if amax is specified
  if(oa is None and da is None):
    # Handles only symmetric subsurface offsets for now
    amin = -amax
    avals = np.linspace(amin,amax,na)
    # Compute angle axis
    da = avals[1] - avals[0]; oa = avals[0]

  # Allocate output and convert to angle
  ext = 4
  angs = np.zeros([nro,nx,na,nz],dtype='float32')
  # Verbosity
  rverb = False; cverb = False
  if(verb):
    if((oro is None or dro is None) and nro > 1):
      rverb = True
    else:
      cverb = True
  # Loop over rho
  for iro in range(nro):
    if(rverb): printprogress("nrho:",iro,nro)
    if(cverb and nro > 1): print("rho=%.3f (%d/%d)"%(oro + iro*dro,iro+1,nro))
    convert2ang(nx,nh,oh,dh,nta,ota,dta,na,oa,da,nz,oz,dz,ext,offt[iro],angs[iro],nthrds,cverb)
  if(rverb): printprogress("nrho:",nro,nro)

  # Transpose and return
  if(transp):
    if(nro > 1):
      return np.ascontiguousarray(np.transpose(angs,(0,2,3,1)))
    else:
      return np.ascontiguousarray(np.transpose(angs[0],(1,2,0)))
  else:
    if(nro > 1):
      return angs
    else:
      return angs[0]

def get_angssk_axis(amax=70,na=281):
  """ Given a maximum angle, returns the angle axis """
  amin = -amax
  avals = np.linspace(amin,amax,na)
  da = avals[1] - avals[0]; oa = avals[0]
  return na,oa,da

def off2angkzx(off,ohx,dhx,dz,ohy=0.0,dhy=1.0,oz=0.0,na=None,amax=60,oa=None,da=None,
               transp=False,eps=1.0,nthrds=4,cverb=False,rverb=False):
  """
  Converts subsurface offset gathers to openeing angle gathers via a
  radial trace transform in the wavenumber domain

  Parameters
    off    - subsurface offset gathers of shape [nro,nhy,nhx,nz,ny,nx] (nro is optional)
    ohx    - origin of subsurface offset axis
    dhx    - sampling of subsurface offset axis
    dz     - sampling in depth
    oz     - depth origin [0.0]
    ohy    - origin of y-subsurface offsets [0.0]
    dhy    - sampling of y-subsurface offsets [1.0]
    na     - number of angles [nhx]
    amax   - maximum angle [60 degrees]
    oa     - origin of angle axis (not needed if na and amax specified)
    da     - spacing on angle axis (not needed if na and amax specified)
    transp - input has shape [nro,nhy,nhx,ny,nx,nz] instead of assumed shape [False]
    eps    - regularization parameter (larger will enforce smoother output) [1.0]
    nthrds - number of OpenMP threads to parallelize over gathers [4]
    rverb  - residual migration verbosity flag [False]
    cverb  - gather conversion verbosity flag [False]

  Returns the converted angle gathers of shape [nro,ny,nx,naz,na,nz]
  """
  # Handle case of 2D residual migration [nro,nhx,nx,nz]
  if(len(off.shape) == 4):
    # Get dimensions
    [nro,nhx,nx,nz] = off.shape; nhy = 1; ny = 1
    off  = off.reshape([nro,nhy,nhx,ny,nx,nz])
    # [nro,nhy,nhx,ny,nx,nz] -> [nro,ny,nx,nz,nhy,nhx]
    offi = np.ascontiguousarray(np.transpose(off,(0,3,4,5,1,2)))
  if(len(off.shape) == 5):
    nro = 1
    if(transp == False):
      # Get dimensions
      [nhy,nhx,nz,ny,nx] = off.shape
      offe = np.expand_dims(off,axis=0)
      # Transpose to correct shape
      # [nro,nhy,nhx,nz,ny,nx] -> [nro,ny,nx,nz,nhy,nhx]
      offi = np.ascontiguousarray(np.transpose(offe,(0,4,5,3,1,2)))
    else:
      # Get dimensions
      [nhy,nhx,ny,nx,nz] = off.shape
      offe = np.expand_dims(off,axis=0)
      # Transpose to correct shape
      # [nro,nhy,nhx,ny,nx,nz] -> [nro,ny,nx,nz,nhy,nhx]
      offi = np.ascontiguousarray(np.transpose(offe,(0,3,4,5,1,2)))

  # Pad input based on desired number of angles
  if(na is None):
    na = nhx
  paddims = [(0,0)]*(offi.ndim-1)
  paddims.append((0,na-nhx))
  offip = np.pad(offi,paddims,mode='constant').astype('complex64')
  ang = np.zeros(offip.shape,dtype='complex64')

  # Compute angle axis
  oa = -amax; da = 2*amax/na

  # Loop over rho
  for iro in progressbar(range(nro),"rho",verb=rverb):
    convert2angkzkykx(nx*ny,                                   # Number of gathers
                      nz,oz,dz,nhy,ohy,dhy,na,ohx,dhx,oa,da,   # Axes of input and output
                      offip[iro],ang[iro],eps,nthrds,cverb)    # Input and output and parameters

  # Transpose, window and return
  if(transp):
    if(nro > 1):
      angw = ang[:,0,:,:,0,:] # [nro,ny,nx,nz,naz,na] -> [nro,nx,nz,na]
      return np.ascontiguousarray(np.transpose(angw,(0,1,3,2))) # [nro,nx,nz,na] -> [nro,nx,na,nz]
    else:
      angw = ang[0]
      ango = np.ascontiguousarray(np.transpose(angw,(0,1,3,4,2))) # [ny,nx,nz,naz,na] -> [ny,nx,naz,na,nz]
  else:
    if(nro > 1):
      angw = ang[:,0,:,:,0,:] # [nro,ny,nx,nz,naz,na] -> [nro,nx,nz,na]
      ango = np.ascontiguousarray(np.transpose(angw,(0,3,2,1))) # [nro,nx,nz,na] -> [nro,na,nz,nx]
    else:
      angw = ang[0]
      ango = np.ascontiguousarray(np.transpose(angw,(3,4,2,0,1))) # [ny,nx,nz,naz,na] -> [naz,na,nz,ny,nx]

  return ango

def get_angkzx_axis(na,amax=60):
  """
  Returns the angle axis

  Parameters
    na   - number of angles
    amax - maximum angle

  Returns na,oa,da
  """
  oa = -amax; da = 2*amax/na
  return na,oa,da

