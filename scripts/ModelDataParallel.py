"""
Models 2D hydrophone data (shot/receiver gathers) in parallel 
(parallelizes over shot using OpenMP) using a scalar 
acoustic wave equation.

All inputs and output files must be in SEPlib format.
While I have the capability of handling both big 
and little endian (xdr_float vs native_float) the code
for now requires big endian.

@author: Joseph Jennings
@version: 2019.11.03
"""
from __future__ import print_function
import sys, os, argparse, configparser
import seppy
import numpy as np
import scaas.scaas2dpy as sca2d
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "bx": 50,
    "bz": 50,
    "alpha": 0.99,
    "dt": 0.004,
    "nsx": 1,
    "osx": 0,
    "dsx": 1,
    "srcz": 0,
    "nrx": None,
    "orx": 0,
    "drx": 1,
    "recz": 0,
    "nthreads": None,
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
parser.add_argument("src=",help="Input source time function (on propagation time grid)")
parser.add_argument("vel=",help="Input velocity model (nz,nx)")
parser.add_argument("out=",help="Output hydrophone data (nt,nrx,nsx)")
# Padding parameters
paddingArgs = parser.add_argument_group('Padding parameters')
paddingArgs.add_argument("-bz",help='Top and bottom padding [50 samples]',type=int)
paddingArgs.add_argument("-bx",help='Left and right [50 samples]',type=int)
paddingArgs.add_argument("-alpha",help='Decay rate of taper [0.99]',type=float)
paddingArgs.add_argument("-tvel",help='Velocity to fill the top after padding [1500 m/s]',type=float)
paddingArgs.add_argument("-z1",help="Depth for filling with tvel velocity [51 samples]",type=int)
# Acquisition parameters
acquisitionArgs = parser.add_argument_group('Acquisition parameters')
acquisitionArgs.add_argument("-nsx",help='Number of sources [1]',type=int)
acquisitionArgs.add_argument("-osx",help='Initial source position [0 sample]',type=int)
acquisitionArgs.add_argument("-dsx",help='Spacing between sources [10 samples]',type=int)
acquisitionArgs.add_argument("-srcz",help='Source depth [0 sample]',type=int)
acquisitionArgs.add_argument("-nrx",help='Number of receivers [all surface points]',type=int)
acquisitionArgs.add_argument("-orx",help='Initial receiver position [0 sample]',type=int)
acquisitionArgs.add_argument("-drx",help='Spacing between receivers [1 samples]',type=int)
acquisitionArgs.add_argument("-recz",help='Receiver depth [0 sample]',type=int)
# Data parameters
dataArgs = parser.add_argument_group('Data parameters')
dataArgs.add_argument("-dt",help='Output data sampling rate step [0.004 s]',type=float)
# Quality check
qcArgs = parser.add_argument_group('QC parameters')
qcArgs.add_argument("-plotacq",help='Plot acquisition (y or [n])')
qcArgs.add_argument("-verb",help="Verbosity flag (y or [n])")
# Other parameters
miscArgs = parser.add_argument_group('Miscellaneous parameters')
miscArgs.add_argument("-nthreads",help='Number of CPU threads to use [nsx]',type=int)
args = parser.parse_args(remaining_argv)

## Get arguments
# Padding
bz = args.bz; bx = args.bx; alpha = args.alpha
tvel = args.tvel; z1 = args.z1

# Acquisition
nsx = args.nsx; osx = args.osx; dsx = args.dsx; srcz = args.srcz
nrx = args.nrx; orx = args.orx; drx = args.drx; recz = args.recz

# Data
dt = args.dt

# Threads
nthreads = args.nthreads

# QC
verb  = args.verb
if(verb == "n"):
  verb = 0
else:
  verb = 1

plotacq = args.plotacq
if(plotacq == "n"):
  plotacq = 0
else:
  plotacq = 1

# Set up SEP
sep = seppy.sep(sys.argv)
# Read in velocity model
vaxes,vel = sep.read_file("vel")
vel = vel.reshape(vaxes.n,order='F')
vel = vel.astype('float32')
# Convert to km/s if needed
nz = vaxes.n[0]; dz = vaxes.d[0];
nx = vaxes.n[1]; dx = vaxes.d[1];
# Read in source
saxes,src = sep.read_file("src")
src = src.astype('float32')
ntu = saxes.n[0]; dtu = saxes.d[0]

# Pad model for absorbing boundaries
velp = np.pad(vel,((bx,bx),(bz,bz)),'edge')
# Pad for laplacian stencil
velp = np.pad(velp,((5,5),(5,5)),'constant')
nxp,nzp = velp.shape

# Set up the acquisition
osx += bx + 5; orx += bx + 5
srcz  += bz + 5; recz  += bz + 5
# Receivers at every gridpoint
if(nrx == None):
  nrx = nx

# Create receiver coordinates
nrec = np.zeros(nrx,dtype='int32') + nrx
allrecx = np.zeros([nsx,nrx],dtype='int32')
allrecz = np.zeros([nsx,nrx],dtype='int32')
# Create all receiver positions
recs = np.linspace(orx,orx + (nrx-1)*drx,nrx)
for isx in range(nsx):
  allrecx[isx,:] = (recs[:]).astype('int32')
  allrecz[isx,:] = np.zeros(len(recs),dtype='int32') + recz

# Create source coordinates
nsrc = np.ones(nsx,dtype='int32')
allsrcx = np.zeros([nsx,1],dtype='int32')
allsrcz = np.zeros([nsx,1],dtype='int32')
# All source x positions in one array
srcs = np.linspace(osx,osx + (nsx-1)*dsx,nsx)
for isx in range(nsx):
  allsrcx[isx,0] = int(srcs[isx])
  allsrcz[isx,0] = int(srcz)

# Crete input wavelet array
allsrcs = np.zeros([nsx,1,ntu],dtype='float32')
for isx in range(nsx):
  allsrcs[isx,0,:] = src[:]

# Create output data array
fact = int(dt/dtu); ntd = int(ntu/fact)
allshot = np.zeros((nsx,ntd,nrx),dtype='float32')

# Set up a wave propagation object
sca = sca2d.scaas2d(ntd,nzp,nxp,dt,dx,dz,dtu,bx,bz,alpha)

# Parallel forward modeling
sca.fwdprop_multishot(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,velp,allshot,nthreads)

## Write out all shots
datout = np.transpose(allshot,(1,2,0))
daxes = seppy.axes([ntd,nrx,nsx],[0.0,(orx-bx)*dx,(osx-bx)*dx],[dt,drx*dx,dsx*dx])
sep.write_file("out",daxes,datout)

