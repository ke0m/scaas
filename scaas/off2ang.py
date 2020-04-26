"""
Functions for converting subsurface offset to angle gathers
@author: Joseph Jennings
@version: 2020.03.24
"""
import numpy as np
from scaas.adcig import convert2ang
from utils.ptyprint import printprogress

def off2ang(off,oh,dh,dz,oz=0.0,na=281,amax=70,oa=None,da=None,nta=601,ota=-3,dta=0.01,
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

def get_ang_axis(amax=70,na=281):
  """ Given a maximum angle, returns the angle axis """
  amin = -amax
  avals = np.linspace(amin,amax,na)
  da = avals[1] - avals[0]; oa = avals[0]
  return na,oa,da

