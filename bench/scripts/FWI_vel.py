"""
Performs FWI of acoustic pressure data using
an LBFGS solver

All inputs and output files must be in SEPlib format.
While I have the capability of handling both big
and little endian (xdr_float vs native_float) the code
for now requires big endian.

@author: Joseph Jennings
@version: 2019.11.26
"""
from __future__ import print_function
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import scaas.fwi as fwi
import opt.nlopt as opt
import opt.optqc as optqc
import matplotlib.pyplot as plt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "plotacq": "n",
    "verb": "n",
    "nthreads": 1,
    "izt": 0,
    "izb": 0
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
parser.add_argument("out=",help="Output estimated model (nz,nx)")
# Inversion movies
movArgs = parser.add_argument_group('Inversion movies')
movArgs.add_argument("-ofn",help="Objective function value")
movArgs.add_argument("-mmov",help="Model iteration movie")
movArgs.add_argument("-gmov",help="Gradient iteration movie")
movArgs.add_argument("-dmov",help="Data iteration movie")
movArgs.add_argument("-wtrials",help="Write trial results (y or [n])")
movArgs.add_argument("-trim",help="Trim the output models and gradients ([y] or n)")
movArgs.add_argument("-sidxs",help="Indices of shots to keep for writing data movies [idx1,idx2,...] (default is all)")
# Other parameters
miscArgs = parser.add_argument_group('Miscellaneous parameters')
miscArgs.add_argument("-nthreads",help='Number of CPU threads to use [1]',type=int)
miscArgs.add_argument("-izt",help="Top depth sample for taper [0]",type=int)
miscArgs.add_argument("-izb",help="Bottom depth sample for taper [0] (default no taper)",type=int)
# Quality check
qcArgs = parser.add_argument_group('QC parameters')
qcArgs.add_argument("-plotacq",help='Plot acquisition (y or [n])')
qcArgs.add_argument("-verb",help="Verbosity flag (y or [n])")
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Number of threads
nthrd = args.nthreads

# Gradient taper
izt = args.izt; izb = args.izb

# Inversion trials
wtrials = sep.yn2zoo(args.wtrials)
trim    = sep.yn2zoo(args.trim)

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
bx = int(padps['bx']); bz = int(padps['bz']); alpha = float(padps['alpha'])
padps['bx'] = bx; padps['bz'] = bz; padps['alpha'] = alpha

# Get source and receiver depths
depthps = sep.from_header("in",['srcz','recz'])
srcz = int(depthps['srcz']); recz = int(depthps['recz'])

# Pad model for absorbing boundaries
velp = np.pad(vel,((bx,bx),(bz,bz)),'edge')
# Pad for laplacian stencil
velp = np.pad(velp,((5,5),(5,5)),'constant')
nzp,nxp = velp.shape

# Set up the acquisition
adict = {}
osxp = osx + bx + 5; orxp = orx + bx + 5
srczp  = srcz + bz + 5; reczp  = recz + bz + 5

# Create receiver coordinates
nrec = np.zeros(nrx,dtype='int32') + nrx
allrecx = np.zeros([nsx,nrx],dtype='int32')
allrecz = np.zeros([nsx,nrx],dtype='int32')
# Create all receiver positions
recs = np.linspace(orxp,orxp + (nrx-1)*drx,nrx)
for isx in range(nsx):
  allrecx[isx,:] = (recs[:]).astype('int32')
  allrecz[isx,:] = np.zeros(len(recs),dtype='int32') + reczp

# Save to dictionary
adict['nrec'] = nrec
adict['allrecx'] = allrecx
adict['allrecz'] = allrecz

# Create source coordinates
nsrc = np.ones(nsx,dtype='int32')
allsrcx = np.zeros([nsx,1],dtype='int32')
allsrcz = np.zeros([nsx,1],dtype='int32')
# All source x positions in one array
srcs = np.linspace(osxp,osxp + (nsx-1)*dsx,nsx)
for isx in range(nsx):
  allsrcx[isx,0] = int(srcs[isx])
  allsrcz[isx,0] = int(srczp)

# Save to dictionary
adict['nsrc'] = nsrc
adict['allsrcx'] = allsrcx
adict['allsrcz'] = allsrcz
adict['nex'] = nsx

if(plotacq):
  vmin = np.min(velp); vmax = np.max(velp)
  plt.figure(1)
  plt.imshow(velp,extent=[0,nxp,nzp,0],vmin=vmin,vmax=vmax,cmap='jet')
  plt.scatter(allrecx[0,:],allrecz[0,:]) # Receivers don't change. Plot all for first shot
  plt.scatter(allsrcx[:,0],allsrcz[:,0]) # Shot does change. Plot each for each shot
  plt.grid()
  plt.show()

# Create input wavelet array
allsrcs = np.zeros([nsx,1,ntu],dtype='float32') # One source for every shot
for isx in range(nsx):
  allsrcs[isx,0,:] = src[:]

## Create FWI object
maxes = seppy.axes([nzp,nxp],[0.0,0.0],[dz,dx])
tpdict = {}; tpdict['izt'] = izt; tpdict['izb'] = izb;
gl2 = fwi.fwi(maxes,saxes,allsrcs,daxes,dat,adict,padps,tpdict,nthrd)

##### Inputs for solver ######
n      = nxp*nzp
m      = np.zeros(1,dtype='int32') + 5
iflag  = np.zeros(1,dtype='int32')
diagco = False; icall = 0

# Objective function value and axis
f = np.zeros(1,dtype='float32')
ofaxes = seppy.axes([1],[0.0],[1.0])

# Create output gradient array
grad = np.zeros(velp.shape,dtype='float32')
diag = np.zeros(n,dtype='float32')

# Create working array
w = np.zeros(n*(2*m[0] + 1) + 2*m[0],dtype='float32')

# Saved state variables
isave = np.zeros(8,dtype='int32')
dsave = np.zeros(14,dtype='float32')

# Get the shot indices for writing
sidxs = sep.read_list(args.sidxs,np.arange(nsx),dtype='int')

# Create the optqc objects
dia = None; diat = None
if(trim):
  # Trim dictionary
  trpars = {}
  trpars['lidx'] = bx+5; trpars['ridx'] = bx+nx+5
  trpars['tidx'] = bz+5; trpars['bidx'] = bz+nz+5
  # Trimmed axes
  tmaxes = seppy.axes([nz,nx],maxes.o,maxes.d)
  tdaxes = seppy.axes([nt,nrx,len(sidxs)],daxes.o,daxes.d)
  dia  = optqc.optqc(sep,"-ofn",ofaxes,"-mmov","-gmov",tmaxes,"-dmov",tdaxes,trpars=trpars)
  if(wtrials):
    diat = optqc.optqc(sep,"-ofn",ofaxes,"-mmov","-gmov",tmaxes,trials=True,trpars=trpars)
else:
  # Trim dictionary (just trim the edges for the laplacian)
  trpars = {}
  trpars['lidx'] = 5; trpars['ridx'] = 2*bx+nx
  trpars['tidx'] = 5; trpars['bidx'] = 2*bz+nz
  tmaxes = seppy.axes([2*bx+nz,2*bz+nz],maxes.o,maxes.d)
  dia  = optqc.optqc(sep,"-ofn",ofaxes,"-mmov","-gmov",tmaxes,"-dmov",daxes,trpars=trpars)
  if(wtrials):
    diat = optqc.optqc(sep,"-ofn",ofaxes,"-mmov","-gmov",tmaxes,trials=True)

# Keep models and gradients from two steps
# This is because the solver will attempt a step length
# that is too large and then use the previous model
mods = []
mods.append(np.zeros(velp.shape,dtype='float32'))
mods.append(np.zeros(velp.shape,dtype='float32'))
mods[1][:] = velp[:]
grds = []
grds.append(np.zeros(velp.shape,dtype='float32'))
grds.append(np.zeros(velp.shape,dtype='float32'))
dats = []
dats.append(np.zeros([nt,nrx,len(sidxs)],dtype='float32'))
dats.append(np.zeros([nt,nrx,len(sidxs)],dtype='float32'))

# Keep track of iterations
itercheck = 0

# Run the inversion
while(True):
  # FWI objective function and gradient evaluation
  f[0] = gl2.gradientL2(velp,grad)

  # Initialize arrays at first iteration
  if(itercheck == 0):
    grds[1][:] = grad[:]
    dats[1][:] = gl2.get_moddat(sidxs)

  # Write trial steps if requested
  if(wtrials):
    diat.outputH(f,velp,grad)

  # Call solver and update model
  opt.lbfgs(n,m,velp,f,grad,diagco,diag,w,iflag,isave,dsave)
  mods[0][:] = mods[1][:]; grds[0][:] = grds[1][:]; dats[0][:] = dats[1][:];
  mods[1][:] = velp[:];    grds[1][:] = grad[:];    dats[1][:] = gl2.get_moddat(sidxs)

  # Check if a new iterate was found (first trial always written)
  if(itercheck != isave[4] or iflag[0] == 0):
    # Update the iteration number
    itercheck = isave[4]
    # Write the new inversion outputs
    dia.outputH(f,mods[0],grds[0],dats[0])

  icall += 1
  if(iflag[0] <= 0 or icall > 500): break

# Write the output model
vel[:,:] = velp[bz+5:bz+5+nz,bx+5:bx+5+nx]
sep.write_file("out",vaxes,vel)

