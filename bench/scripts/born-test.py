"""
Tests the forward Born scattering

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
dvel = np.zeros([nz,nx],dtype='float32')
dvel[49,49] = 1.0;

# Pad the velocity model
bx = 50; bz = 50
velp  = np.pad(vel, ((bx,bz),(bx,bz)),'edge')
dvelp = np.pad(dvel,((bx,bz),(bx,bz)),'edge')

# Pad for laplacian
velp  = np.pad(velp, ((5,5),(5,5)),'constant')
dvelp = np.pad(dvelp,((5,5),(5,5)),'constant')

# Total velocity
veltp = velp + dvelp

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
nzp,nxp = velp.shape; alpha = 0.99
sca = sca2d.scaas2d(ntd,nxp,nzp,dtd,dx,dz,dtu,bx,bz,alpha)

# Output data arrays
hdat = np.zeros([ntd,nrx],dtype='float32')
tdat = np.zeros([ntd,nrx],dtype='float32')
bdat = np.zeros([ntd,nrx],dtype='float32')

# Difference between two forwards
sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,velp ,hdat)
sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,veltp,tdat)
ddat = tdat - hdat
# Born modeling 
sca.brnfwd_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,velp,dvelp,bdat)

# Plot result
dmax = np.max(ddat); dmin = np.min(ddat)
f,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(ddat,cmap='gray',aspect=2,extent=[0,2,2,0],vmin=dmin,vmax=dmax)
ax[0].set_ylabel('Time (s)',fontsize=14)
ax[0].set_xlabel('x (km)',fontsize=14)
ax[1].imshow(bdat,cmap='gray',aspect=2,extent=[0,2,2,0],vmin=dmin,vmax=dmax)
ax[1].set_yticks([])
ax[1].set_xlabel('x (km)',fontsize=14)
plt.subplots_adjust(wspace=-0.5)

f1 = plt.figure()
plt.imshow(ddat-bdat,cmap='gray',aspect=2,extent=[0,2,2,0])
plt.colorbar()

f2 = plt.figure()
t = np.linspace(0,(ntd-1)*dtd,ntd)
plt.plot(t,ddat[:,49],label='Difference')
plt.plot(t,bdat[:,49],label='Born')
plt.legend()

plt.show()
