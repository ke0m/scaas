"""
Tests the allocated memory for my FWI class
and gradient calculation 

@author: Joseph Jennings
@version: 2019.11.25
"""
from __future__ import print_function
import numpy as np
import inpout.seppy as seppy
import scaas.scaas2dpy as sca2d
import scaas.fwi as fwi
from scaas.wavelet import ricker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

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

# Observed data
dattru = np.zeros([1,ntd,1],dtype='float32')

# Model the data
sca.fwdprop_oneshot(fsrc,srcx,srcz,1,recx,recz,1,vtrup,dattru)

## Create the FWI object
# Axes
maxes = seppy.axes([nzp,nxp],[0.0,0.0],[dz,dx])
saxes = seppy.axes([ntu],[0.0],[dtu])
daxes = seppy.axes([ntd,1,1],[0.0,0.0,0.0],[dtd,1.0,1.0])
# Dictionaries
adict = {}
adict['nsrc'] = np.asarray([1],dtype='int32'); adict['allsrcx'] = srcx; adict['allsrcz'] = srcz
adict['nrec'] = np.asarray([1],dtype='int32'); adict['allrecx'] = recx; adict['allrecz'] = recz
adict['nex']  = 1;
pdict = {}
pdict['bz'] = bz; pdict['bx'] = bx; pdict['alpha'] = alpha
# Create object
gl2 = fwi.fwi(maxes,saxes,fsrc,daxes,dattru,adict,pdict,1)

# Compute gradient
# Loop over gradient calculation
for itry in range(1000):
  grad = np.zeros([nzp,nxp],dtype='float32')
  f = gl2.gradientL2(vmodp,grad)

print(f)
# Plot gradient
plt.imshow(grad)
plt.show()


