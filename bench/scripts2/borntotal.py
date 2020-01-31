"""
Forward born for multiple shots

@author: Joseph Jennings
@version: 2019.12.12
"""
from __future__ import print_function
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import scaas.scaas2dpy as sca2d
from scaas.wavelet import ricker
import matplotlib.pyplot as plt

# Create the time axis
ntu = 2000; otu = 0.0; dtu = 0.001;
ntd = 2000; otd = 0.0; dtd = 0.001;

# Create the spatial axes
nz = 100; oz = 0.0; dz = 10.0;
nx = 100; ox = 0.0; dx = 10.0;

# Create the velocity model
velval = 2000.0
vel = np.zeros([nz,nx],dtype='float32') + velval 
dvel = np.zeros([nz,nx],dtype='float32')
dvel[49,49] = 1.0;

# Pad the velocity model
bx = 50; bz = 50
velp  = np.pad(vel, ((bx,bz),(bx,bz)),'edge')
dvelp = np.pad(dvel,((bx,bz),(bx,bz)),'edge')

# Pad for laplacian
velp  = np.pad(velp, ((5,5),(5,5)),'constant')
dvelp = np.pad(dvelp,((5,5),(5,5)),'constant')

# Get padded lengths
nxp,nzp = velp.shape

# Create source coordinates
nsx = int(nx/10)+1; osxp = bx + 5; dsx = 10
srczp = bz + 5
nsrc = np.ones(nsx,dtype='int32')
allsrcx = np.zeros([nsx,1],dtype='int32')
allsrcz = np.zeros([nsx,1],dtype='int32')
# All source x positions in one array
srcs = np.linspace(osxp,osxp + (nsx-1)*dsx,nsx)
for isx in range(nsx):
  allsrcx[isx,0] = int(srcs[isx])
  allsrcz[isx,0] = int(srczp)

# Create receiver coordinates
nrx = nx; orxp = bx + 5; drx = 1
reczp = bz + 5
nrec = np.zeros(nrx,dtype='int32') + nrx 
allrecx = np.zeros([nsx,nrx],dtype='int32')
allrecz = np.zeros([nsx,nrx],dtype='int32')
# Create all receiver positions
recs = np.linspace(orxp,orxp + (nrx-1)*drx,nrx)
for isx in range(nsx):
  allrecx[isx,:] = (recs[:]).astype('int32')
  allrecz[isx,:] = np.zeros(len(recs),dtype='int32') + reczp

# Plot acqusition
plt.figure(1)
# Plot velocity model
vmin = np.min(vel); vmax = np.max(vel)
plt.imshow(velp,extent=[0,nxp,nzp,0],vmin=vmin,vmax=vmax,cmap='jet')
# Get all source positions
plt.scatter(allrecx[0,:],allrecz[0,:])
plt.scatter(allsrcx[:,0],allsrcz[:,0])
plt.show()

# Create the wavelet array
freq = 20; amp = 100.0; dly = 0.2;
fsrc = ricker(ntu,dtu,freq,amp,dly)
allsrcs = np.zeros([nsx,1,ntu],dtype='float32')
for isx in range(nsx):
  allsrcs[isx,0,:] = fsrc[:]

# Create output data array
fact = int(dtd/dtu); ntd = int(ntu/fact)
ddat = np.zeros((nsx,ntd,nrx),dtype='float32')

# Set up a wave propagation object
sca = sca2d.scaas2d(ntd,nxp,nzp,dtd,dx,dz,dtu,bx,bz,alpha=0.99)

# Forward modeling for all shots
nthreads = 2
sca.brnfwd(allsrcs,allsrcx,allsrcz,nsrc,allrecx,allrecz,nrec,nsx,velp,dvelp,ddat,nthreads)
sep = seppy.sep([])
ddatt = np.transpose(ddat,(1,2,0))
daxes = seppy.axes([ntd,nrx,nsx],[0.0,0.0,0.0],[dtu,1,1])
greyargs = "gainpanel=a wantscalebar=y"
sep.pltgrey(daxes,ddatt,greyargs)

