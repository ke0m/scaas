import inpout.seppy as seppy
import numpy as np
from scaas.wavelet import ricker
import oway.defaultgeomnode as geom
from  scaas.trismooth import smooth
import matplotlib.pyplot as plt
from utils.plot import plot_wavelet
import time

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
freq = 8; amp = 0.5; dly = 0.2;
wav = ricker(n1,d1,freq,amp,dly)

osx = 150; dsx = 50
wei = geom.defaultgeomnode(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                           nsx=2,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

# Create the frequency domain source
wfft = wei.fft1(wav,d1,minf=1.0,maxf=31.0)

dchunks = wei.create_mod_chunks(2,wfft)

wei.set_model_pars(velin,ref,ntx=15,px=112,nthrds=4,wverb=True)

for ichnk in dchunks:
  wei.model_chunk(ichnk['nsrc'],ichnk['srcy'],ichnk['srcx'],ichnk['wav'],
                  ichnk['nrec'],ichnk['recy'],ichnk['recx'],ichnk['dat'])

nw,ow,dw = wei.get_freq_axis()

plt.figure()
plt.imshow(np.real(dchunks[0]['dat']).T,cmap='gray',interpolation='sinc')
plt.figure()
plt.imshow(np.real(dchunks[1]['dat']).T,cmap='gray',interpolation='sinc')
plt.show()

#sep.write_file("mycmplxdat.H",chnk1['dat'].T,os=[0,0,ow,0,0],ds=[dx,dy,dw,1.0,1.0])
#sep.write_file("mydat.H",dat.T,os=[0,0,0,osx,0],ds=[d1,dx,dy,dsx,1.0])

#plt.figure(figsize=(10,10))
#plt.imshow(np.real(dat[0,0,:,0,:]),cmap='gray',interpolation='sinc')
#plt.figure(figsize=(10,10))
#plt.imshow(np.real(dat[0,1,:,0,:]),cmap='gray',interpolation='sinc')
#plt.show()

