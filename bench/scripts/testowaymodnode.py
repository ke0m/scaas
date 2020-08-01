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
freq = 8; amp = 0.5; dly = 0.2; it0 = int(dly/d1)
wav = ricker(n1,d1,freq,amp,dly)

osx = 150; dsx = 50
wei = geom.defaultgeomnode(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                           nsx=2,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

# Create the frequency domain source
wfft = wei.fft1(wav,d1,minf=1.0,maxf=31.0)

dchunks = wei.create_mod_chunks(1,wfft)

wei.set_model_pars(velin,refsm,ntx=15,px=112,nthrds=4,wverb=True)

dats = []
for ichnk in dchunks:
  dats.append(wei.model_chunk(ichnk))

nw,ow,dw = wei.get_ofreq_axis()

odat = np.squeeze(np.asarray(dats))

# Inverse FFT
odatt = wei.ifft1(odat,nw,ow,dw,n1,it0=it0)

datreg = wei.make_sht_cube(odatt)

plt.figure()
plt.imshow(datreg[0].T,cmap='gray',interpolation='sinc')
plt.figure()
plt.imshow(datreg[1].T,cmap='gray',interpolation='sinc')
plt.show()

#sep.write_file("mycmplxdat.H",chnk1['dat'].T,os=[0,0,ow,0,0],ds=[dx,dy,dw,1.0,1.0])
sep.write_file("mydatnode.H",datreg.T,os=[0,0,osx],ds=[d1,dx,dsx])

#plt.figure(figsize=(10,10))
#plt.imshow(np.real(dat[0,0,:,0,:]),cmap='gray',interpolation='sinc')
#plt.figure(figsize=(10,10))
#plt.imshow(np.real(dat[0,1,:,0,:]),cmap='gray',interpolation='sinc')
#plt.show()

