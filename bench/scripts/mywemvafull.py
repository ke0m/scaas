import inpout.seppy as seppy
import numpy as np
from scaas.wavelet import ricker
import oway.defaultgeom as geom
from scaas.trismooth import smooth
from genutils.plot import plot_img2d, plot_vel2d
from genutils.movie import viewimgframeskey

# IO
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

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 8; amp = 0.5; dly = 0.2
wav = ricker(n1,d1,freq,amp,dly)

osx = 200; dsx = 20; nsx = 20
wei = geom.defaultgeom(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

#wei.plot_acq(mod=velin)

dat = wei.model_data(wav,d1,dly,minf=1.0,maxf=31.0,vel=velin,ref=refsm,time=True,ntx=15,px=112,
                     nthrds=4,sverb=True)

dslo = np.ones(ref.shape,dtype='complex64')*5e-05
dimg = wei.fwemva(dslo,dat,d1,minf=1.0,maxf=31.0,vel=velin,nthrds=4,sverb=True)
dslo = wei.awemva(dimg,dat,d1,minf=1.0,maxf=31.0,vel=velin,nthrds=4,sverb=True)

plot_img2d(dimg[:,0,:],show=False)
plot_img2d(dslo[:,0,:],cmap='jet')

