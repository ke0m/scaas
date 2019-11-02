## Plots acquisition geometry for modeling

import sys, argparse, ConfigParser
import seppy
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "vmin": 0.0,
    "vmax": 0.0,
    "verb": 'y',
    "klean": 'y',
    }   
if args.conf_file:
  config = ConfigParser.SafeConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("Defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.set_defaults(**defaults)
# Input files
parser.add_argument("vel=",help="Input velocity")
# Required arguments
requiredNamed = parser.add_argument_group('required parameters')
requiredNamed.add_argument("-nrx",help="Number of receivers",required=True,type=int)
requiredNamed.add_argument("-orx",help="Receiver origin",required=True,type=int)
requiredNamed.add_argument("-drx",help="Receiver spacing",required=True,type=int)
requiredNamed.add_argument("-recz",help="Receiver depth",required=True,type=int)
requiredNamed.add_argument("-nsx",help="Number of sources",required=True,type=int)
requiredNamed.add_argument("-osx",help="Source origin",required=True,type=int)
requiredNamed.add_argument("-dsx",help="Source spacing",required=True,type=int)
requiredNamed.add_argument("-srcz",help="Source depth",required=True,type=int)
requiredNamed.add_argument("-bz",help="Z padding",required=True,type=int)
requiredNamed.add_argument("-bx",help="X padding",required=True,type=int)
# Optional arguments
parser.add_argument("-verb",help="Verbosity flag ([y] or n)")
parser.add_argument("-klean",help="Clean intermediate files ([y] or n)")
parser.add_argument("-vmin",help="Minimum velocity for plotting (default is actual min vel)",type=float)
parser.add_argument("-vmax",help="Maximum velocity for plotting (default is actual min vel)",type=float)
parser.add_argument("-qc",help="Plot a QC of a homogeneous velocity model ([y] or n)",type=str)
args = parser.parse_args(remaining_argv)

nrx = args.nrx; nsx = args.nsx
orx = args.orx; osx = args.osx
drx = args.drx; dsx = args.dsx
recz = args.recz; srcz = args.srcz
bz = args.bz; bx = args.bx

vmin = args.vmin; vmax = args.vmax

verb  = args.verb
qc = args.qc
clean = args.klean
if(verb == "n"):
  verb = 0 
else:
  verb = 1 
if(qc == "n"):
  qc = 0
else:
  qc = 1
if(clean == 'n'):
  clean = 0
else:
  clean = 1

# Set up SEP
sep = seppy.sep(sys.argv)

# Read in velocity model
vaxes, vel = sep.read_file("vel")
vel = vel.reshape(vaxes.n,order='F')
nz = vaxes.n[0]; nx = vaxes.n[1]

# Need two paddings to account for the bug
# in my padvel.cpp code
velp = np.pad(vel,((5,5),(5,5)),'constant') 
print(velp.shape)

# Surround with boundary color
velp[0:bz,:] = -1000.0;
velp[:,0:bx] = -1000.0;
velp[vel.shape[0]-(bz)+10:,:] = -1000.0;
velp[:,vel.shape[1]-(bx)+10:] = -1000.0;

velp[0:5,:] = 0.0;
velp[:,0:5] = 0.0;
velp[velp.shape[0]-5:,:] = 0.0;
velp[:,velp.shape[1]-5:] = 0.0;

# Create receiver grid
if(nrx != 1):
  #recs = np.arange(orx,orx + (nrx-1)*drx,drx)
  recs = np.linspace(orx,orx + (nrx-1)*drx,nrx)
  recdepth = np.zeros(len(recs)) + recz
  if(verb): print("Number of receivers=%d"%(len(recs)))
else:
  recs = orx
  recdepth = recz
if(verb): print(recs)

# Create source grid
if(nsx !=1):
  #srcs = np.arange(osx,osx + (nsx-1)*dsx,dsx)
  srcs = np.linspace(osx,osx + (nsx-1)*dsx,nsx)
  srcdepth = np.zeros(len(srcs)) + srcz
  if(verb): print("Number of sources=%d"%(len(srcs)))
else:
  srcs = osx
  srcdepth = srcz
if(verb): print(srcs)

if(qc):
  homoone = np.zeros([nz,nx]) + 1.0
  sep.write_file(None,vaxes,homoone,'homoqc.H')
  tappad = "./Bin/taperpad-test.x bx=%d bz=%d < homoqc.H > homoqctap.H \
      upout=homoqctappad.H"%(bx,bz)
  sp = subprocess.check_call(tappad,shell=True)
  paxes,tappad = sep.read_file(None,ifname='homoqctappad.H')
  tappad = tappad.reshape(paxes.n,order='F')
  if(clean): 
    sp = subprocess.check_call('Rm homoqc.H homoqctap.H homoqctappad.H',shell=True)

# Plot
if(vmin == 0.0):
  vmin = np.min(velp)
if(vmax == 0.0):
  vmax = np.max(velp)
plt.figure(1)
plt.imshow(velp,extent=[0,nx+10,nz+10,0],vmin=vmin,vmax=vmax)
plt.scatter(recs,recdepth)
plt.scatter(srcs,srcdepth)
plt.grid()
plt.figure(2)
plt.imshow(tappad,extent=[0,nx+10,nz+10,0],vmin=0.95,vmax=1.0)
plt.scatter(recs,recdepth)
plt.scatter(srcs,srcdepth)
plt.grid()
plt.show()

