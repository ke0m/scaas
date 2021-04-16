"""
Applies linear or hyperbolic mutes
to gathers

@author: Joseph Jennings
@version: 2020.07.11
"""
import numpy as np
from oway.mutter import muteall

def mute(data,dt,dx,ot=0.0,ox=0.0,t0=0,tp=0.150,v0=1.45,x0=0.0,slope0=None,slopep=None,
         half=True,absx=True,inner=False,hyper=False):
  """
  Applies linear or hyperbolic 2D mutes to gathers

  Data is smoothly weighted inside the mute zone
  The weight is zero for t <      (x-x0) * slope0
  The weight is one  for t > tp + (x-x0) * slopep

  The signs are reversed for inner=True

  Parameters:
    data   - input data [ngather,nx,nt]
    dt     - temporal sampling of data
    dx     - spatial sampling of data
    ot     - time axis origin [0.0]
    ox     - x-axis orgin [0.0]
    t0     - starting time [0.0]
    tp     - end time [0.15]
    v0     - velocity [1.45]
    x0     - starting space [0.0]
    slope0 - slope [1/v0]
    slopep - end slope [slope0]
    half   - the second axis is half-offset instad of full [True]
    absx   - use absolute value |x-x0| [True]
    inner  - apply an inner mute [False]
    hyper  - do a hyperbolic mute [False]
  """
  # Get dimensions
  dshape = data.shape
  if(len(dshape) == 2):
    din = np.expand_dims(data,axis=0).astype('float32')
  else:
    din = data.astype('float32')
  [ng,nx,nt] = din.shape

  if(slope0 is None):
    slope0 = 1/v0
  if(slopep is None):
    slopep = slope0

  dout = np.zeros(din.shape,dtype='float32')
  # Apply mute
  muteall(ng,nx,ox,dx,nt,ot,dt,      # Sizes
          tp,t0,v0,slope0,slopep,x0, # Mute parameters
          absx,inner,hyper,half,     # Flags
          din,dout)                  # Input and output

  return dout
  
