import inpout.seppy as seppy
import numpy as np
from scaas.wavelet import ricker
import oway.defaultgeom as geom
from  scaas.trismooth import smooth
import matplotlib.pyplot as plt
from genutils.plot import plot_wavelet, plot_dat2d, plot_img2d
from genutils.movie import viewcube3d
import time

sep = seppy.sep()

# Dimensions
nx = 500; dx = 0.015
ny = 500; dy = 0.015
nz = 400; dz = 0.005

# Build input slowness
vz = 1.5 +  np.linspace(0.0,dz*(nz-1),nz)
velx  = np.ascontiguousarray(np.repeat(vz[np.newaxis,:],nx,axis=0))
vel = np.ascontiguousarray(np.repeat(velx[np.newaxis,:,:],ny,axis=0)).T

#viewcube3d(vel,ds=[dz,dy,dx],cmap='jet',cbar=True)

velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:] = vel[:]

# Build reflectivity
ref = np.zeros(velin.shape,dtype='float32')
ref[349,49:449,49:449] = 1.0
npts = 25
refsm = smooth(smooth(smooth(ref,rect1=npts,rect2=npts),rect1=npts,rect2=npts),rect1=npts,rect2=npts)

#viewcube3d(refsm,ds=[dz,dy,dx],vmin=-1,vmax=1)

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 8; amp = 0.5; dly = 0.2;
wav = ricker(n1,d1,freq,amp,dly)

osx,dsx = 250,50
osy,dsy = 250,50
wei = geom.defaultgeom(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=1,dsx=dsx,osx=osx,nsy=1,dsy=dsy,osy=osy)

#wei.plot_acq(velin)

beg = time.time()
dat = wei.model_data(wav,d1,dly,minf=1.0,maxf=31.0,vel=velin,ref=refsm,
                     ntx=15,px=112,nty=15,py=112,nthrds=40,wverb=True)

img = wei.image_data(dat,d1,minf=1.0,maxf=31.0,vel=velin,nthrds=40,wverb=True)

print("Elapsed=%f"%(time.time()-beg))
viewcube3d(dat[0,0].T,show=False)
viewcube3d(img)
#
##plot_dat2d(dat[0,1,0],dt=d1,show=False)
##plot_img2d(img[:,0,:],dx=dx,dz=dz)

