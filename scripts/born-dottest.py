"""
Tests the linearized forward and 
adjoint propagation

@author: Joseph Jennings
@version: 2019.12.08
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
bx = 50; bz = 50
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
fsrc = np.random.rand(ntu).astype('float32')

# Create wave propagation object
nzp,nxp = velp.shape; alpha = 0.97
sca = sca2d.scaas2d(ntd,nxp,nzp,dtd,dx,dz,dtu,bx,bz,alpha)

# Random inputs and temporary arrays
x  = np.random.rand(nxp,nzp).astype('float32')
xh = np.zeros(velp.shape,dtype='float32')
y  = np.random.rand(ntd,nrx).astype('float32')
yh = np.zeros([ntd,nrx],dtype='float32')

# Born modeling 
sca.brnfwd_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,velp,x,yh)

# Born adjoint
sca.brnadj_oneshot(fsrc,srcx,srcz,1,recx,recz,nrx,velp,xh,y)

dotx = np.dot(x.flatten(),xh.flatten())
doty = np.dot(y.flatten(),yh.flatten())

print("%f %f"%(dotx,doty))

