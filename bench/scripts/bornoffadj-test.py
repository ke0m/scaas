"""
Tests the extended linearized adjoint

@author: Joseph Jennings
@version: 2019.12.09
"""
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

# Output data array
bdat = np.zeros([ntd,nrx],dtype='float32')

# Born modeling 
sca.brnfwd_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,velp,dvelp,bdat)

# Create the extended image
zidx = 5
rnh = 2*zidx + 1
dvele = np.zeros([rnh,nzp,nxp],dtype='float32')
# Born adjoint
sca.brnoffadj_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,velp,rnh,dvele,bdat)

dvelewind = dvele[:,bz+5:nz+bz+5,bx+5:nx+bx+5]
imin = np.min(dvelewind[zidx,:,:]); imax = np.max(dvelewind[zidx,:,:])

fig,ax1 = plt.subplots(3,zidx)
for i in range(zidx):
  # Top row
  ax1[0,i].imshow(dvelewind[i,:,:],extent=[0,2,0,2],cmap='gray',vmin=imin,vmax=imax)
  ax1[0,i].set_yticks([])
  ax1[0,i].set_xticks([])
  if(i != int(zidx/2)):
    ax1[1,i].axis('off')
  else:
    ax1[1,i].imshow(dvelewind[zidx,:,:],extent=[0,2,2,0],cmap='gray',vmin=imin,vmax=imax)
  # Bottom row  
  ax1[2,i].imshow(dvelewind[zidx+i,:,:],extent=[0,2,0,2],cmap='gray',vmin=imin,vmax=imax)
  ax1[2,i].set_yticks([])
  ax1[2,i].set_xticks([])

plt.show()

