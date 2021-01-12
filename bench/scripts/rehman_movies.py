import inpout.seppy as seppy
import numpy as np
import scaas.defaultgeom as geom
import scaas.scaas2dpy as sca
from scaas.wavelet import ricker
from scaas.trismooth import smooth
import matplotlib.pyplot as plt
from utils.plot import plot_wavelet
from utils.movie import viewimgframeskey

sep = seppy.sep()

vaxes,vel = sep.read_file("/home/joe/phd/projects/widb/Dat/marmvel.H")
vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F')).astype('float32')
[nz,nx] = vaxes.n; [oz,ox] = vaxes.o; [dz,dx] = vaxes.d

# Resample the model
j1 = 3; j2 = 3
vel1   = smooth(vel,rect1=2,rect2=2)
velr   = vel1[::j2,::j1]
velrsm = smooth(velr,rect1=10,rect2=10)

[nzr,nxr] = velr.shape; dzr = dz*j2; dxr = dx*j1

nsx = 2; osx=int(nxr/2); dsx = 1; bx = 50; bz = 50
prp = geom.defaultgeom(nxr,dxr,nzr,dzr,nsx=nsx,osx=osx,dsx=dsx,bx=bx,bz=bz)

prp.plot_acq(velrsm,cmap='jet',zcut=51,tvel=1500.0)

# Make the wavelet
ntu = 4000; dtu = 0.001
freq = 20; amp = 100.0; dly=0.2
wav = ricker(ntu,dtu,freq,amp,dly)
plot_wavelet(wav,dtu)

# Model the wavefields
#twfld = prp.model_fwdwfld(velr  ,wav)
swfld = prp.model_fwdwfld(velrsm,wav)

#print(np.min(twfld),np.max(twfld))
#print(np.min(swfld),np.max(swfld))

#viewimgframeskey(twfld,interp='bilinear',transp=False,vmin=-5,vmax=5,cmap='seismic',wbox=14,hbox=5)
#viewimgframeskey(swfld,interp='bilinear',transp=False,vmin=-5,vmax=5,cmap='seismic',wbox=14,hbox=5)

# Model the data and compute the residual

dat = prp.model_fulldata(velr  ,wav,dtd=0.001,nthrds=2,verb=True)
mod = prp.model_fulldata(velrsm,wav,dtd=0.001,verb=True)

asrc = mod - dat

# plt.figure()
# plt.imshow(dat[0],cmap='gray',interpolation='sinc',vmin=-1,vmax=1)
# plt.figure()
# plt.imshow(mod[0],cmap='gray',interpolation='sinc',vmin=-1,vmax=1)
# plt.figure()
# plt.imshow(mod[0]-dat[0],cmap='gray',interpolation='sinc')
#plt.show()

awfld = prp.model_adjwfld(velrsm,asrc)
#print(np.min(awfld),np.max(awfld))
#viewimgframeskey(awfld,interp='bilinear',transp=False,cmap='seismic',wbox=14,hbox=5,pclip=0.01)

img = np.sum(swfld*awfld,axis=0)

plt.figure()
plt.imshow(img,cmap='gray')
plt.show()