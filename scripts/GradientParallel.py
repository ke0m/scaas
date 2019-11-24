"""
Computes an FWI gradient for all shots in parallel

All inputs and output files must be in SEPlib format.
While I have the capability of handling both big
and little endian (xdr_float vs native_float) the code
for now requires big endian.

@author: Joseph Jennings
@version: 2019.11.21
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
    "nthreads": None,
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
# Other parameters
miscArgs = parser.add_argument_group('Miscellaneous parameters')
miscArgs.add_argument("-nthreads",help='Number of CPU threads to use [nsx]',type=int)
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
nrec = np.zeros(nrx,dtype='int32') + nrx 
allrecx = np.zeros([nsx,nrx],dtype='int32')
allrecz = np.zeros([nsx,nrx],dtype='int32')
# Create all receiver positions
recs = np.linspace(orxp,orxp + (nrx-1)*drx,nrx)
for isx in range(nsx):
  allrecx[isx,:] = (recs[:]).astype('int32')
  allrecz[isx,:] = np.zeros(len(recs),dtype='int32') + reczp

# Create source coordinates
nsrc = np.ones(nsx,dtype='int32')
allsrcx = np.zeros([nsx,1],dtype='int32')
allsrcz = np.zeros([nsx,1],dtype='int32')
# All source x positions in one array
srcs = np.linspace(osxp,osxp + (nsx-1)*dsx,nsx)
for isx in range(nsx):
  allsrcx[isx,0] = int(srcs[isx])
  allsrcz[isx,0] = int(srczp)

if(plotacq):
  vmin = np.min(velp); vmax = np.max(velp)
  plt.figure(1)
  plt.imshow(velp,extent=[0,nxp,nzp,0],vmin=vmin,vmax=vmax,cmap='jet')
  plt.scatter(recs,recdepth)
  plt.scatter(srcs,srcdepth)
  plt.grid()
  plt.show()

# Create input wavelet array
allsrcs = np.zeros([nsx,1,ntu],dtype='float32')
for isx in range(nsx):
  allsrcs[isx,0,:] = src[:]

# Create output data array
fact = int(dt/dtu); ntd = int(ntu/fact)
moddat = np.zeros((nsx,ntd,nrx),dtype='float32')

# Create output gradient arrays
igrad = np.zeros(velp.shape,dtype='float32')
grad = np.zeros(velp.shape,dtype='float32')

# Set up a wave propagation object
sca = sca2d.scaas2d(ntd,nxp,nzp,dt,dx,dz,dtu,bx,bz,alpha)

# Forward modeling for all shots
sca.fwdprop_multishot(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,velp,moddat,nthreads)

# Compute adjoint source
asrc = -(moddat - dat)

f,axarr = plt.subplots(1,2)
axarr[0].imshow(moddat[10,:,:])
axarr[1].imshow(dat[10,:,:])
plt.show()

# Gradient for all shots
sca.gradient_multishot(allsrcs,allsrcx,allsrcz,nsrc,asrc,allrecx,allrecz,nrec,nsx,velp,grad,nthreads)

# TODO: Unpad the gradient
upgrad = np.zeros([nz,nx],dtype='float32')
upgrad[:,:] = grad[bz+5:bz+5+nz,bx+5:bx+5+nx]

## Write the gradient
paxes = seppy.axes([nzp,nxp],[0.0,0.0],[dz,dx])
sep.write_file("out",paxes,grad)

## Write the synthetic data if desired
if(sep.get_fname("-moddat") != None):
  sep.write_file("-moddat",daxes,moddat)

