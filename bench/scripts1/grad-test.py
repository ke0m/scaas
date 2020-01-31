"""
A simple test for the gradient. Analogous to the 
grad-test.cpp code I wrote a year ago

@author: Joseph Jennings
@version: 2019.11.16
"""
from __future__ import print_function
import numpy as np
import inpout.seppy as seppy
import scaas.scaas2dpy as sca2d
from wavelet import ricker
import matplotlib.pyplot as plt

sep = seppy.sep([])

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
freq = 20; amp = 10.0; dly = 0.2;
fsrc = ricker(ntu,dtu,freq,amp,dly)
_,ofsrc= sep.read_file(None,"/home/joe/phd/internships/ti/presentation/ricker.H")
ofsrcc = ofsrc.astype('float32')

fsrcrse = np.zeros(ntd,dtype='float32')
kt = 0
for it in range(ntu):
  if(it%skip == 0):
    fsrcrse[kt] = fsrc[it]
    kt += 1

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

# Model the wavefield
wfld = np.zeros([ntd,nzp,nxp],dtype='float32')
sca.fwdprop_wfld(fsrc,srcx,srcz,1,vmodp,wfld)

# Second time derivative of wavefield
d2pt = np.zeros([ntd,nzp,nxp],dtype='float32')
sca.d2t(wfld,d2pt)

# Laplacian of wavefield
d2px = np.zeros([ntd,nzp,nxp],dtype='float32')
sca.d2x(wfld,d2px)

# Other laplacian of wavefield
lapwfld = np.zeros([ntd,nzp,nxp],dtype='float32')
sca.fwdprop_lapwfld(fsrc,srcx,srcz,1,vmodp,lapwfld)

#it = 300
#plt.figure()
#plt.imshow(d2px[it,:,:])
#plt.colorbar()
#
#plt.figure()
#plt.imshow(lapwfld[it,:,:])
#plt.colorbar()
#
#plt.figure()
#plt.imshow(lapwfld[it,:,:] - d2px[it,:,:])
#plt.show()

# Model the data
sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,1,vtrup,dattru)
sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,1,vmodp,datmod)

#_,owfld = sep.read_file(None,"/home/joe/phd/internships/ti/presentation/fwdwfld.H")
#owfld = owfld.reshape([nzp,nxp,ntd],order='F')

#plt.figure()
#plt.plot(dattru)
#plt.plot(datmod)

# Calculate adjoint source
asrc = (dattru - datmod)
plt.plot(asrc)
plt.show()

# Compare adjoint wavefields
#lsol = np.zeros([ntd,nzp,nxp],dtype='float32')
#print("Outer adjoint")
#sca.adjprop_wfld(asrc,recx,recz,1,vmodp,lsol)
#
## Old gradient
#grad1 = np.zeros([nzp,nxp],dtype='float32')
#sca.calc_grad_d2t(d2pt,lsol,vmodp,grad1)
#
## New gradient
#grad2 = np.zeros([nzp,nxp],dtype='float32')
##sca.calc_grad_d2x(d2px,lsol,fsrcrse,srcx,srcz,1,vmodp,grad2)
#sca.calc_grad_d2x(lapwfld,lsol,fsrcrse,srcx,srcz,1,vmodp,grad2)

# Read in adjoint wavefield
#_,awfld = sep.read_file(None,"/home/joe/phd/internships/ti/presentation/adjwfld.H")
#awfld = awfld.reshape([nzp,nxp,ntd],order='F')

# Calculate gradient
grad = np.zeros([nzp,nxp],dtype='float32')
sca.gradient_oneshot(fsrc,srcx,srcz,1,asrc,recx,recz,1,vmodp,grad)

# Read in the adjoint source and other gradient
#_,odattru = sep.read_file(None,"/home/joe/phd/internships/ti/presentation/dattrut.H")
#_,odatmod = sep.read_file(None,"/home/joe/phd/internships/ti/presentation/datmodt.H")
#oasrc = odattru - odatmod

#_,ograd = sep.read_file(None,"/home/joe/phd/internships/ti/presentation/gradtestint.H")
#ograd = ograd.reshape([nzp,nxp],order='F')

_,fdgrad = sep.read_file(None,"/home/joe/phd/internships/ti/presentation/fdgradtest.H")
fdgrad = fdgrad.reshape([nzp,nxp],order='F')

#grad[srcz,srcx] = ograd[srcz,srcx]
#plt.figure()
#plt.imshow(grad2)
#plt.colorbar()

#plt.figure()
#plt.imshow(grad1)
#plt.colorbar()

plt.figure()
plt.imshow(grad)
plt.colorbar()
plt.show()
