"""
Compares three ways to compute the FWI gradient for 
acoustic wavespeed:
  1) Laplacian
  2) Second time derivative
  3) Finite differences

The first two use the adjoint method and the
same forward and adjoint wave propagation code,
just different methods of computing the gradient.

The third method uses only the forward but
takes quite a few minutes (6401 wave
propagations)

@author: Joseph Jennings
@version: 2019.11.17
"""
from __future__ import print_function
import numpy as np
import inpout.seppy as seppy
import scaas.scaas2dpy as sca2d
from wavelet import ricker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def objf(datmod,dattru):
  """ FWI L2 Objective function """
  res = datmod - dattru
  return 0.5*np.dot(res,res)

# Create the time axis
ntu = 1000; otu = 0.0; dtu = 0.001;
ntd = 1000; otd = 0.0; dtd = 0.001;

# Create the spatial axes
nz = 60; oz = 60.0; dz = 10.0;
nx = 60; ox = 60.0; dx = 10.0;

# Source position
srcx = np.zeros(1,dtype='int32')
srcz = np.zeros(1,dtype='int32')
srcx[0] = 40; srcz[0] = 25

# Receiver position
recx = np.zeros(1,dtype='int32')
recz = np.zeros(1,dtype='int32')
recx[0] = 40; recz[0] = 50;

# Create the wavelet
freq = 20; amp = 20.0; dly = 0.2;
fsrc = ricker(ntu,dtu,freq,amp,dly)

# Create the velocity model
vtruval = 2000.0
vtru = np.zeros([nz,nx],dtype='float32') + vtruval
vmodval = 2001.0
vmod = np.zeros([nz,nx],dtype='float32') + vmodval

# Pad the velocity
px = 5; pz = 5;
vtrup = np.pad(vtru,((px,px),(pz,pz)),'edge')
vmodp = np.pad(vmod,((px,px),(pz,pz)),'edge')

# Pad for laplacian
vtrup = np.pad(vtrup,((5,5),(5,5)),'constant')
vmodp = np.pad(vmodp,((5,5),(5,5)),'constant')

# Create wave propagation object
nzp,nxp = vtrup.shape; 
bx = 25; bz = 25; alpha = 0.97
sca = sca2d.scaas2d(ntd,nxp,nzp,dtd,dx,dz,dtu,bx,bz,alpha)

# Output data arrays
dattru = np.zeros(ntd,dtype='float32')
datmod = np.zeros(ntd,dtype='float32')

# Model the data
sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,1,vtrup,dattru)
sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,1,vmodp,datmod)

# Model the wavefield
wfld = np.zeros([ntd,nzp,nxp],dtype='float32')
sca.fwdprop_wfld(fsrc,srcx,srcz,1,vmodp,wfld)

# Second time derivative of wavefield
d2pt = np.zeros([ntd,nzp,nxp],dtype='float32')
sca.d2t(wfld,d2pt)

# Calculate adjoint source
asrc = (dattru - datmod)

# Calculate adjoint wavefield
lsol = np.zeros([ntd,nzp,nxp],dtype='float32')
sca.adjprop_wfld(asrc,recx,recz,1,vmodp,lsol)

## Gradient with second time derivative
graddt = np.zeros([nzp,nxp],dtype='float32')
sca.calc_grad_d2t(d2pt,lsol,vmodp,graddt)

## Gradient with laplacian
graddx = np.zeros([nzp,nxp],dtype='float32')
sca.gradient_oneshot(fsrc,srcx,srcz,1,asrc,recx,recz,1,vmodp,graddx)

## Gradient with finite difference
# Initial objective function calculation
mis0 = objf(datmod,dattru)

# Velocity perturbation size
dvel = 10.0
vptbp  = np.zeros(vtrup.shape,dtype='float32')
gradfd = np.zeros(vtrup.shape,dtype='float32')
datptb = np.zeros(ntd,dtype='float32')

# Calculate objective function perturbation for each point in the model
for iz in range(5,nzp):
  for ix in range(5,nxp):
    vptbp[:] = vmodp[:]
    vptbp[iz,ix] += dvel
    sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,1,vptbp,datptb)
    misptb = objf(datptb,dattru)
    gradfd[iz,ix] = (misptb - mis0)/dvel

#sep = seppy.sep([])
#gaxes = seppy.axes([nzp,nxp],[0.0,0.0],[dz,dx])
#sep.write_file(None,gaxes,gradfd,'pygradfd1.H')

#gaxes,gradfd = sep.read_file(None,ifname='pygradfd1.H')
#gradfd = gradfd.reshape(gaxes.n,order='F')

cmin = np.min(gradfd); cmax = np.max(gradfd)

fsize=14
# Plot gradients
fig, axarr = plt.subplots(2,3, figsize=(10,5))
# First row
im1 = axarr[0,0].imshow(graddt,extent=[0,0.8,0.8,0],vmin=cmin,vmax=cmax,cmap='jet')
axarr[0,0].set_title('Time',fontsize=fsize)
axarr[0,0].set_ylabel('Y [km]',fontsize=fsize)
axarr[0,0].set_xticks([])
axarr[0,0].tick_params(labelsize=fsize)
im2 = axarr[0,1].imshow(graddx,extent=[0,0.8,0.8,0],vmin=cmin,vmax=cmax,cmap='jet')
axarr[0,1].set_title('Space',fontsize=fsize)
axarr[0,1].set_xticks([])
axarr[0,1].set_yticks([])
im3 = axarr[0,2].imshow(gradfd,extent=[0,0.8,0.8,0],vmin=cmin,vmax=cmax,cmap='jet')
axarr[0,2].set_title('FD',fontsize=fsize)
axarr[0,2].set_yticks([])
axarr[0,2].tick_params(labelsize=fsize)
axarr[0,2].set_xlabel('X [km]',fontsize=fsize)
fig.subplots_adjust(right=0.8)
cbar_ax1 = fig.add_axes([0.82,0.5,0.02,0.4])
fig.colorbar(im3,cax=cbar_ax1)
# Second row
axarr[1,0].imshow(graddt-gradfd,extent=[0,0.8,0.8,0],vmin=cmin,vmax=cmax,cmap='jet')
axarr[1,0].set_title('Time - FD: %g'%(np.linalg.norm(graddt-gradfd)),fontsize=fsize)
axarr[1,0].set_xlabel('X [km]',fontsize=fsize)
axarr[1,0].set_ylabel('Y [km]',fontsize=fsize)
axarr[1,0].tick_params(labelsize=fsize)
axarr[1,1].imshow(graddx-gradfd,extent=[0,0.8,0.8,0],vmin=cmin,vmax=cmax,cmap='jet')
axarr[1,1].set_title('Space - FD: %g'%(np.linalg.norm(graddx-gradfd)),fontsize=fsize)
axarr[1,1].set_xlabel('X [km]',fontsize=fsize)
axarr[1,1].tick_params(labelsize=fsize)
axarr[1,1].set_yticks([])
axarr[1,2].set_visible(False)

plt.show()
