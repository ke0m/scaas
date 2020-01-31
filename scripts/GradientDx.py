"""
Computes an FWI gradient on a shot by shot basis

All inputs and output files must be in SEPlib format.
While I have the capability of handling both big
and little endian (xdr_float vs native_float) the code
for now requires big endian.

@author: Joseph Jennings
@version: 2019.11.12
"""
from __future__ import print_function
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import scaas.scaas2dpy as sca2d
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "plotacq": "n",
    "verb": "n",
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.set_defaults(**defaults)
# Input files
parser.add_argument("in=",help="Input observed data (nt,nrx,nsx)")
parser.add_argument("src=",help="Input source time function (on propagation time grid)")
parser.add_argument("vel=",help="Input smoothed velocity model (nz,nx)")
parser.add_argument("-moddat=",help="Output modeled data (nt,nrx,nsx)")
parser.add_argument("out=",help="Output gradient (nz,nx)")
# Quality check
qcArgs = parser.add_argument_group('QC parameters')
qcArgs.add_argument("-plotacq",help='Plot acquisition (y or [n])')
qcArgs.add_argument("-verb",help="Verbosity flag (y or [n])")
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# QC
verb    = sep.yn2zoo(args.verb)
plotacq = sep.yn2zoo(args.plotacq)

# Read in data
daxes,dat = sep.read_file("in")
dat = dat.reshape(daxes.n,order='F')
nt  = daxes.n[0]; ot  = daxes.o[0]; dt  = daxes.d[0]
nrx = daxes.n[1]; orx = daxes.o[1]; drx = daxes.d[1]
nsx = daxes.n[2]; osx = daxes.o[2]; dsx = daxes.d[2]
# Transpose data
dat = np.ascontiguousarray(np.transpose(dat,(2,0,1)))
dat = dat.astype('float32')

# Read in velocity model
vaxes,vel = sep.read_file("vel")
vel = vel.reshape(vaxes.n,order='F')
vel = vel.astype('float32')
nz = vaxes.n[0]; dz = vaxes.d[0];
nx = vaxes.n[1]; dx = vaxes.d[1];

# Read in source
saxes,src = sep.read_file("src")
src = src.astype('float32')
ntu = saxes.n[0]; dtu = saxes.d[0]

# Get padding parameters
padps = sep.from_header("in",['bx','bz','alpha'])
bx = int(padps[0]); bz = int(padps[1]); alpha = float(padps[2])

# Get source and receiver depths
depthps = sep.from_header("in",['srcz','recz'])
srcz = int(depthps[0]); recz = int(depthps[0])

# Pad model for absorbing boundaries
velp = np.pad(vel,((bx,bx),(bz,bz)),'edge')
# Pad for laplacian stencil
velp = np.pad(velp,((5,5),(5,5)),'constant')
nzp,nxp = velp.shape

# Set up the acquisition
osxp = osx + bx + 5; orxp = orx + bx + 5
srczp  = srcz + bz + 5; reczp  = recz + bz + 5
# Receivers at every gridpoint
if(nrx == None):
  nrx = nx

# Create receiver coordinates
if(nrx != 1):
  recs = np.linspace(orxp,orxp + (nrx-1)*drx,nrx)
  # Assumes a fixed receiver depth
  recdepth = np.zeros(len(recs)) + reczp
else:
  recs = np.zeros(1)
  recs[0] = orxp
  recdepth = reczp
# Convert to int
recs = recs.astype('int32')
recdepth = recdepth.astype('int32')
if(verb): print("Final receiver position: %d"%(recs[-1]))

# Create source coordinates
if(nsx != 1):
  srcs = np.linspace(osxp,osxp + (nsx-1)*dsx,nsx)
  # Assumes a fixed source depth
  srcdepth = np.zeros(len(srcs)) + srczp
else:
  srcdepth = np.zeros(1); srcdepth[0] = srczp
  srcs = np.zeros(1); srcs[0] = osxp
# Convert to int
srcs = srcs.astype('int32')
srcdepth = srcdepth.astype('int32')
if(verb): print("Final source position: %d"%(srcs[-1]))

if(plotacq):
  vmin = np.min(velp); vmax = np.max(velp)
  plt.figure(1)
  plt.imshow(velp,extent=[0,nxp,nzp,0],vmin=vmin,vmax=vmax,cmap='jet')
  plt.scatter(recs,recdepth)
  plt.scatter(srcs,srcdepth)
  plt.grid()
  plt.show()

# Create output data array
fact = int(dt/dtu); ntd = int(ntu/fact)
oneshotmod = np.zeros((ntd,nrx),dtype='float32')
allshotmod = np.zeros((nsx,ntd,nrx),dtype='float32')

# Create output gradient arrays
igrad = np.zeros(velp.shape,dtype='float32')
grad = np.zeros(velp.shape,dtype='float32')

# Set up a wave propagation object
sca = sca2d.scaas2d(ntd,nxp,nzp,dt,dx,dz,dtu,bx,bz,alpha)

# Source coordinates
isrcx = np.zeros(1,dtype='int32'); isrcz = np.zeros(1,dtype='int32')
## Calculate gradient for each shot
for isx in range(nsx):
  print("Shot %d"%(isx))
  # Initialize temporary gradient
  igrad[:] = 0.0
  # Create source coordinates
  isrcx[0] = srcs[isx]; isrcz[0] = srcdepth[isx]
  # Forward modeling for one shot
  sca.fwdprop_oneshot(src,isrcx,isrcz,1,recs,recdepth,nrx,velp,oneshotmod)
  # Save the modeled data
  allshotmod[isx,:,:] = oneshotmod
  #f2,axarr2 = plt.subplots(1,2)
  #axarr2[0].imshow(oneshotmod,cmap='gray');
  #axarr2[1].imshow(dat[isx,:,:],cmap='gray');
  #plt.show()
  # Calculate adjoint source
  asrc = -(oneshotmod - dat[isx,:,:])
  # Calculate gradient for this shot
  sca.gradient_oneshot(src,isrcx,isrcz,1,asrc,recs,recdepth,nrx,velp,igrad)
  # Sum over all gradients
  grad += igrad
  # Plot
  #f1,axarr1 = plt.subplots(1,2)
  #axarr1[0].imshow(grad)
  #axarr1[1].imshow(igrad)
  #plt.show()

# TODO: Unpad the gradient
upgrad = np.zeros([nz,nx],dtype='float32')
upgrad[:,:] = grad[bz+5:bz+5+nz,bx+5:bx+5+nx]

## Write the gradient
paxes = seppy.axes([nzp,nxp],[0.0,0.0],[dz,dx])
sep.write_file("out",paxes,grad)

## Write the synthetic data if desired
if(sep.get_fname("-moddat") != None):
  sep.write_file("-moddat",daxes,allshotmod)


