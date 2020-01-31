"""
Computes the finite difference approximation
of the gradient (no adjoint)

@author: Joseph Jennings
@version: 2019.11.17
"""
from __future__ import print_function
import numpy as np
import inpout.seppy as seppy
import scaas.scaas2dpy as sca2d
from wavelet import ricker
import matplotlib.pyplot as plt

def objf(datmod,dattru):
  res = datmod - dattru
  return 0.5*np.dot(res,res)

# Create the time axis
ntu = 1000; otu = 0.0; dtu = 0.001;
ntd = 1000; otd = 0.0; dtd = 0.001;
skip = int(ntu/ntd)

# Create the spatial axes
nz = 60; oz = 60.0; dz = 10.0;
nx = 60; ox = 60.0; dx = 10.0;

# Source position
srcx = np.zeros(1,dtype='int32')
srcz = np.zeros(1,dtype='int32')
srcx[0] = 40; srcz[0] = 25

# Receiver p
recx = np.zeros(1,dtype='int32')
recz = np.zeros(1,dtype='int32')
recx[0] = 40; recz[0] = 50;

# Create the wavelet
freq = 20; amp = 1.0; dly = 0.2;
fsrc = ricker(ntu,dtu,freq,amp,dly)*dx*dz

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
datptb = np.zeros(ntd,dtype='float32')

# Initial objective function calculation
sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,1,vtrup,dattru)
sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,1,vmodp,datmod)
mis0 = objf(datmod,dattru)
#print("Mis0=%f"%(mis0))

dvel = 10.0
vptbp  = np.zeros(vtrup.shape,dtype='float32')
gradfd = np.zeros(vtrup.shape,dtype='float32')

for iz in range(5,nzp):
  for ix in range(5,nxp):
    vptbp[:] = vmodp[:]
    vptbp[iz,ix] += dvel
    sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,1,vptbp,datptb)
    misptb = objf(datptb,dattru)
    gradfd[iz,ix] = (misptb - mis0)/dvel
    #if(misptb != 0.0):
    #  print("iz=%d ix=%d mis0=%g misptb=%g grad=%g"%(iz,ix,mis0,misptb,gradfd[iz,ix]))

sep = seppy.sep([])
_,fdgrad = sep.read_file(None,"/home/joe/phd/internships/ti/presentation/fdgradtest.H")
fdgrad = fdgrad.reshape([nzp,nxp],order='F')

plt.figure()
plt.imshow(gradfd)
plt.colorbar()

plt.figure()
plt.imshow(fdgrad)
plt.colorbar()

plt.show()

