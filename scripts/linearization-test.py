"""
Linearization test of the scalar acoustic
wave equation

@author: Joseph Jennings
@version: 2019.12.07
"""
from __future__ import print_function
import numpy as np
import inpout.seppy as seppy
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

# Pad the velocity model
bx = 25; bz = 25
velp  = np.pad(vel, ((bx,bz),(bx,bz)),'edge')

# Pad for laplacian
velp  = np.pad(velp, ((5,5),(5,5)),'constant')

# Source position
sxpos = 50; szpos = 0
srcx = np.zeros(1,dtype='int32')
srcz = np.zeros(1,dtype='int32')
srcx[0] = sxpos + bx + 5; srcz[0] = szpos + bz + 5

# Receiver positions
nrx = nx; orx = bx + 5; drx = 1
recx = (np.linspace(orx,orx + (nrx-1)*drx,nrx)).astype('int32')
recz = np.zeros(nrx,dtype='int32') + bz + 5

# Create the wavelet
freq = 20; amp = 100.0; dly = 0.2;
fsrc = ricker(ntu,dtu,freq,amp,dly)

# Create wave propagation object
nzp,nxp = velp.shape; alpha = 0.97
sca = sca2d.scaas2d(ntd,nxp,nzp,dtd,dx,dz,dtu,bx,bz,alpha)

# Output data arrays
hdat = np.zeros([ntd,nrx],dtype='float32')
tdat = np.zeros([ntd,nrx],dtype='float32')
bdat = np.zeros([ntd,nrx],dtype='float32')

# Create perturbation values
nptb = 201; optb = 1.0; dptb = 1.0
ptbs = np.linspace(optb,optb+(nptb-1)*dptb,nptb)
errs = np.zeros(nptb)

# Loop over a range of perturbations
dvxpos = 49; dvzpos = 49
for iptb in range(nptb):
  # Create velocity model and perturbation
  dvelp = np.zeros(velp.shape,dtype='float32')
  dvelp[bx+5+dvxpos,bz+5+dvzpos] = ptbs[iptb] 
  veltp = dvelp + velp
  # Linearized modeling
  sca.brnfwd_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,velp,dvelp,bdat)
  #plt.figure(); plt.imshow(bdat,extent=[0,2,2,0],cmap='gray',aspect=2); plt.colorbar(); plt.show()
  # Two full modelings
  sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,velp ,hdat)
  sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,veltp,tdat)
  ddat = tdat - hdat
  #plt.figure(); plt.imshow(ddat,extent=[0,2,2,0],cmap='gray',aspect=2); plt.colorbar(); plt.show()
  # 2-norm of error
  errs[iptb] = np.sqrt(np.dot(ddat.flatten(),ddat.flatten()))
  print("%f/%f err=%f"%(ptbs[iptb],velval,errs[iptb]))

plt.plot(ptbs,errs)
plt.show()
