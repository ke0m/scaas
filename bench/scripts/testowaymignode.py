import inpout.seppy as seppy
import numpy as np
import oway.defaultgeomnode as geom
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey
import time

sep = seppy.sep()

# Read in the data
sep = seppy.sep()

daxes,dat = sep.read_file("mydatnode.H")
dat = dat.reshape(daxes.n,order='F')
[nt,nx,nsx] = daxes.n; [dt,dx,dsx] = daxes.d; [ot,ox,osx] = daxes.o

# Prepare the data
datt = dat.T
datin = np.ascontiguousarray(datt.reshape([1,nsx,1,nx,nt]))

# Dimensions
nx = 800; dx = 0.015
ny = 1;   dy = 0.125
nz = 400; dz = 0.005

# Build input slowness
vz = 1.5 +  np.linspace(0.0,dz*(nz-1),nz)
vel = np.ascontiguousarray(np.repeat(vz[np.newaxis,:],nx,axis=0).T)

velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel[:]

wei = geom.defaultgeomnode(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                           nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

# Reshape the data
datinr = datin.reshape([nsx*nx,nt])

# FFT the data
datw = wei.fft1(datinr,dt,minf=1.0,maxf=31.0)

# FFT the wavelet
wav  = np.zeros(nt,dtype='float32')
wav[0] = 1.0
wavw = wei.fft1(wav,dt,minf=1.0,maxf=31.0)

# Chunk the data
dchunks = wei.create_img_chunks(1,wavw,datw)

# Set the imaging pars
wei.set_image_pars(velin,nhx=20,ntx=15,nthrds=4,wverb=True)

# Image each chunk
imgl = []
for ichnk in dchunks:
  imgl.append(wei.image_chunk(ichnk))

# Sum all chunks
imgs = np.sum(np.asarray(imgl),axis=0)

viewimgframeskey(imgs[0,:,:,0],transp=False,cmap='gray')

#plt.imshow(imgs[:,0,:],cmap='gray')
#plt.show()

#img = wei.image_data(datin,dt,minf=1.0,maxf=31.0,vel=velin,nhx=20,ntx=15,
#                     nthrds=4,wverb=True,eps=0.0)

#nhx,ohx,dhx = wei.get_off_axis()

#imgt = np.transpose(img,(2,4,3,1,0)) # [nhy,nhx,nz,ny,nx] -> [nz,nx,ny,nhx,nhy]

#sep.write_file("myimgext.H",imgt,os=[0,0,0,ohx,0],ds=[dz,dy,dx,dhx,1.0])

#sep.write_file("myimg.H",img,ds=[dz,dy,dx])

