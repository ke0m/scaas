import inpout.seppy as seppy
import numpy as np
from scaas.wavelet import ricker
import oway.defaultgeom as geom
from oway.utils import fft1, phzshft
from scaas.trismooth import smooth
from genutils.plot import plot_img2d, plot_dat2d, plot_vel2d
from genutils.movie import viewcube3d

# IO
sep = seppy.sep()

# Dimensions
nx = 500; dx = 0.025
ny = 1;   dy = 0.025
nz = 1000; dz = 0.005

# Build input slowness
vz = 1.5 +  np.linspace(0.0,dz*(nz-1),nz)
vel = np.ascontiguousarray(np.repeat(vz[np.newaxis,:],nx,axis=0).T)

# Read in F3 velocity
vaxes,vel = sep.read_file("/home/joe/phd/projects/resfoc/bench/f3/dat/miglintz5mwind.H")
vel = np.transpose(vel.reshape(vaxes.n,order='F'),(0,2,1)).astype('float32')
vel = np.ascontiguousarray(vel)

velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel[:,200,:]
plot_vel2d(vel[:,200,:],dz=dz,dx=dx)

# Build reflectivity
ref = np.zeros(velin.shape,dtype='float32')
ref[649,0,49:449] = 1.0 
npts = 25
refsm = smooth(smooth(smooth(ref,rect1=npts),rect1=npts),rect1=npts)

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 8; amp = 0.5; dly = 0.2
wav = ricker(n1,d1,freq,amp,dly)

osx = 250; dsx = 10; nsx = 1
wei = geom.defaultgeom(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

dat = wei.model_data(wav,d1,dly,minf=1.0,maxf=31.0,vel=velin,ref=refsm,time=True,ntx=15,px=112,
                     nrmax=20,nthrds=4,sverb=False,wverb=True,eps=0.0)

img = wei.image_data(dat,d1,minf=1.0,maxf=31.0,vel=velin,nrmax=20,nthrds=4,sverb=False,wverb=True)

plot_dat2d(dat[0,0,0],show=False)
plot_img2d(img[:,0,:])


