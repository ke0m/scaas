import inpout.seppy as seppy
import numpy as np
from scaas.wavelet import ricker
from scaas.trismooth import smooth
from oway.utils import fft1, ifft1, phzshft
from oway.tutorial import ssr3tut
from genutils.plot import plot_img2d, plot_vel2d, plot_dat2d
from genutils.movie import viewimgframeskey
import matplotlib.pyplot as plt

sep = seppy.sep()

# Dimensions
nx = 500; dx = 0.025
ny = 1;   dy = 0.025
nz = 1000; dz = 0.005

# Build input slowness
vz = 1.5 +  np.linspace(0.0,dz*(nz-1),nz)
vel = np.ascontiguousarray(np.repeat(vz[np.newaxis,:],nx,axis=0).T)

#velin = np.zeros([nz,ny,nx],dtype='float32')
#velin[:,0,:] = vel[:,:]

# Read in F3 velocity
vaxes,vel = sep.read_file("/home/joe/phd/projects/resfoc/bench/f3/dat/miglintz5mwind.H")
vel = np.transpose(vel.reshape(vaxes.n,order='F'),(0,2,1)).astype('float32')
vel = np.ascontiguousarray(vel)

velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel[:,200,:]
plot_vel2d(vel[:,200,:],dz=dz,dx=dx)

# Build reflectivity
ref = np.zeros(velin.shape,dtype='float32')
ref[749,0,49:449] = 1.0 
npts = 25
refsm = smooth(smooth(smooth(ref,rect1=npts),rect1=npts),rect1=npts)

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 8; amp = 0.5; dly = 0.2 
wav = ricker(n1,d1,freq,amp,dly)

minf,maxf = 1.0,31
jf = 1
# Create the input frequency domain source and get original frequency axis
nwo,ow,dw,wfft = fft1(wav,d1,minf=minf,maxf=maxf)
wfftd = wfft[::jf]
nwc = wfftd.shape[0] # Get the number of frequencies to compute
dwc = jf*dw

print("Frequency axis: nw=%d ow=%f dw=%f"%(nwc,ow,dwc))

ntx,px = 15,112
# Single square root object
ssf = ssr3tut(nx ,ny ,nz ,
              dx ,dy ,dz ,
              nwc,ow ,dwc,
              ntx=ntx,px=px,nrmax=20)

ssf.set_slows(1/velin)

# Allocate output data (surface wavefield)
datw = np.zeros([nwc,ny,nx],dtype='complex64')

# Allocate the source for one shot
sou = np.zeros([nwc,ny,nx],dtype='complex64')
sou[:,0,250] = wfftd[:]
datw = ssf.mod_allw(ref,sou,True)
datwt = datw.T

# Phase shift
datp = phzshft(datwt,ow,dwc,dly).T
plt.figure(); plt.imshow(np.real(datp[:,0,:]),cmap='gray'); plt.show()

# Inverse FFT
datwt = np.transpose(datw,(1,2,0)) # [nw,ny,nx] -> [ny,nx,nw]
datt = ifft1(datwt,nwo,ow,dw,n1,int(dly/d1))
plt.figure(); plt.imshow(datt[0,:,:].T,cmap='gray'); plt.show()

# Make impulsive wavelet at source position
imp = np.zeros([nwc,ny,nx],dtype='complex64')
imp[:,0,250] = np.complex(1.0,0.0)

# Migration
img = ssf.mig_allw(datp,imp)

plt.imshow(img[:,0,:],cmap='gray'); plt.show()


