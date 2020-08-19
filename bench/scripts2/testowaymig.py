import inpout.seppy as seppy
import numpy as np
import oway.defaultgeom as geom
from  scaas.trismooth import smooth
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey
import time

sep = seppy.sep()

# Read in the data
sep = seppy.sep()

daxes,dat = sep.read_file("mydat.H")
dat = dat.reshape(daxes.n,order='F')
[nt,nx,ny,nsx] = daxes.n; [dt,dx,dy,dsx] = daxes.d; [ot,ox,oy,osx] = daxes.o

# Prepare the data
datt = dat.T
datin = np.ascontiguousarray(datt.reshape([1,nsx,ny,nx,nt]))

# Dimensions
nx = 800; dx = 0.015
ny = 1;   dy = 0.125
nz = 400; dz = 0.005

# Build input slowness
vz = 1.5 +  np.linspace(0.0,dz*(nz-1),nz)
vel = np.ascontiguousarray(np.repeat(vz[np.newaxis,:],nx,axis=0).T)

velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel[:]

wei = geom.defaultgeom(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                       nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

beg = time.time()
img = wei.image_data(datin,dt,minf=1.0,maxf=31.0,vel=velin,nhx=20,ntx=15,
                     nthrds=24,wverb=False,eps=0.0)
print("Elapsed=%f"%(time.time()-beg))

imgang = wei.to_angle(img,transp=True)
print(imgang.shape)

#viewimgframeskey(img[0,:,:,0,:],transp=False)
viewimgframeskey(imgang[0,:,0,:,:],transp=True)

#plt.figure(); plt.imshow(img[:,0,:],cmap='gray',interpolation='sinc'); plt.show()

#sep.write_file("myimg.H",img,[
