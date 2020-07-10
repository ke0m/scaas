import inpout.seppy as seppy
import numpy as np
from oway.ssr3 import ssr3
import oway.defaultgeom as geom
from  scaas.trismooth import smooth
import matplotlib.pyplot as plt

sep = seppy.sep()

# Dimensions
nx = 800; dx = 0.015
ny = 1;   dy = 0.125
nz = 400; dz = 0.005

# Build input slowness
vz = 1.5 +  np.linspace(0.0,dz*(nz-1),nz)
vel = np.ascontiguousarray(np.repeat(vz[np.newaxis,:],nx,axis=0).T)

velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel[:]

# Build reflectivity
ref = np.zeros(velin.shape,dtype='float32')
ref[349,0,49:749] = 1.0
npts = 25
refsm = smooth(smooth(smooth(ref,rect1=npts),rect1=npts),rect1=npts)

# Read in time domain wavelet
waxes,wav = sep.read_file("../oway/src/srmodmig/wav.rsf",form='native')
[n1] = waxes.n; [d1] = waxes.d
t0 = 50*d1

osx = 300; dsx = 100
wei = geom.defaultgeom(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=3,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

dat = wei.model_data(wav,d1,t0,minf=1.0,maxf=31.0,vel=velin,ref=refsm,time=True,ntx=15,px=112)

nw,ow,dw = wei.get_freq_axis()

#sep.write_file("mycmplxdat.H",dat.T,os=[0,0,ow,0,0],ds=[dx,dy,dw,1.0,1.0])
sep.write_file("mydat.H",dat.T,os=[0,0,0,osx,0],ds=[d1,dx,dy,dsx,1.0])

#plt.figure(figsize=(10,10))
#plt.imshow(np.real(dat[0,0,:,0,:]),cmap='gray',interpolation='sinc')
#plt.figure(figsize=(10,10))
#plt.imshow(np.real(dat[0,1,:,0,:]),cmap='gray',interpolation='sinc')
#plt.show()

