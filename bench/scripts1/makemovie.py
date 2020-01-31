"""
Models 2D pressure wavefield from a point source using a scalar
acoustic wave equation. Mostly useful for debugging, and
display purposes

All inputs and output files must be in SEPlib format.
While I have the capability of handling both big 
and little endian (xdr_float vs native_float) the code
for now requires big endian.

@author: Joseph Jennings
@version: 2019.11.11
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
    "srcx": 0,
    "srcz": 0,
    "plotacq": "n",
    "verb": "n",
    "figdir": None,
    "lapwfld": "n"
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
parser.add_argument("vel=",help="Input velocity model (nx,nz)")
parser.add_argument("out=",help="Output wavefield (nz,nx,nt)")
# Padding parameters
paddingArgs = parser.add_argument_group('Padding parameters')
paddingArgs.add_argument("-bz",help='Top and bottom padding [50 samples]',type=int)
paddingArgs.add_argument("-bx",help='Left and right [50 samples]',type=int)
paddingArgs.add_argument("-alpha",help='Decay rate of taper [0.99]',type=float)
paddingArgs.add_argument("-tvel",help='Velocity to fill the top after padding [1500 m/s]',type=float)
paddingArgs.add_argument("-z1",help="Depth for filling with tvel velocity [51 samples]",type=int)
# Acquisition parameters
acquisitionArgs = parser.add_argument_group('Acquisition parameters')
acquisitionArgs.add_argument("-srcx",help='Initial source position [0 sample]',type=int)
acquisitionArgs.add_argument("-srcz",help='Source depth [0 sample]',type=int)
# Quality check
qcArgs = parser.add_argument_group('QC parameters')
qcArgs.add_argument("-plotacq",help='Plot acquisition (y or [n])')
qcArgs.add_argument("-verb",help="Verbosity flag (y or [n])")
qcArgs.add_argument("-figdir",help="Wavefield plotted on velocity model figures [None]",type=str)
# Other arguments
otherArgs = parser.add_argument_group('Other parameters')
otherArgs.add_argument("-lapwfld",help='Gives the laplacian of th wavefield (y or [n])',type=str)
otherArgs.add_argument("-dt",help="Time sampling of the output wavefield [0.004]",type=float)
args = parser.parse_args(remaining_argv)

## Get arguments
# Padding
bz = args.bz; bx = args.bx; alpha = args.alpha
tvel = args.tvel; z1 = args.z1

# Acquisition
srcx = args.srcx; srcz = args.srcz

# Data
dt = args.dt

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

lapwfld = args.lapwfld
if(lapwfld == "n"):
  lapwfld = 0
else:
  lapwfld = 1

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
nzp,nxp = velp.shape

# Set up the acquisition
srcxp = srcx + bx + 5; 
srczp = srcz + bz + 5;

srcdepth = np.zeros(1); srcdepth[0] = srczp
srcs = np.zeros(1); srcs[0] = srcxp
# Convert to int
srcs = srcs.astype('int32')
srcdepth = srcdepth.astype('int32')
if(verb): print("Final source position: %d"%(srcs[-1]))

if(plotacq):
  vmin = np.min(velp); vmax = np.max(velp)
  plt.figure(1)
  plt.imshow(velp,extent=[0,nxp,nzp,0],vmin=vmin,vmax=vmax,cmap='jet')
  plt.scatter(srcs,srcdepth)
  plt.grid()
  plt.show()

# Create output data array
fact = int(dt/dtu); ntd = int(ntu/fact)

# Set up a wave propagation object
sca = sca2d.scaas2d(ntd,nxp,nzp,dt,dx,dz,dtu,bx,bz,alpha)

# Source coordinates
isrcx = np.zeros(1,dtype='int32'); isrcz = np.zeros(1,dtype='int32')

## Forward modeling
# Create the wavefield
onewfld = np.zeros((ntd,nzp,nxp),dtype='float32')
# Forward modeling for one shot
if(lapwfld):
  sca.fwdprop_lapwfld(src,srcs,srcdepth,1,velp,onewfld)
else:
  sca.fwdprop_wfld(src,srcs,srcdepth,1,velp,onewfld)

# make fake receivers
drx = 30
recs = np.linspace(10*drx/1000.0,10*drx/1000.0+(nx-1)*drx/1000.0,(nx/30))
recz = np.zeros(len(recs)) + srcz;


# Loop over wavefield and make figures
for it in range(ntd):
  fig = plt.figure(figsize=(10,5))
  ax = fig.add_subplot(111)
  plt.scatter((srcx)*dx/1000.0,(srcz+5)*dx/1000.0,c='tab:red',marker='*',s=100)
  plt.scatter((recs),(recz+5)*dx/1000.0,c='tab:green',marker='v',s=50)
  plt.imshow(onewfld[it,bz:nz+bz,bx+5:nx+bx+5],extent=[0,((nx-1)*dx)/1000.0,((nz-1)*dz)/1000.0,0],vmin=-0.05,vmax=0.05,cmap='gray')
  im = plt.imshow(velp[bz:nz+bz,bx+5:nx+bx+5]/1000.0,extent=[0,(nx-1)*dx/1000,(nz-1)*dz/1000.0,0],vmin=1500.0/1000.0,vmax=vmax/1000.0,alpha=0.2,cmap='jet')
  ax.set_xlabel('x (km)', fontsize=14)
  ax.set_ylabel('z (km)', fontsize=14)
  cbar = plt.colorbar(im, orientation="horizontal", pad=0.2)
  cbar.set_alpha(1.0)
  cbar.set_label('km/s',fontsize=12)
  cbar.draw_all()
  plt.savefig('./fig/slice%d.png'%it,bbox_inches='tight',transparent=True)
  #plt.show()

