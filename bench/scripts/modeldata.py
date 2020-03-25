import scaas.defaultgeom as geom
from scaas.wavelet import bandpass
from scaas.trismooth import smooth
from deeplearn.utils import resample
import numpy as np
import wget
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey
from utils.plot import plot_wavelet

# Download the velocity models
#url = "http://sep.stanford.edu/data/media/public/sep/joseph29/vels.npy"
#vels = np.load(wget.download(url))
vels = np.load('vels.npy')

#viewimgframeskey(vels,cmap='jet',interp='bilinear')

vel = np.ascontiguousarray(vels[3,:,:].T)

#plt.imshow(vel,cmap='jet',interpolation='bilinear')
#plt.show()

# Resample the velocity model
nzo = 200; nxo = 400; dx = 20; dz = 20
velsm,ds = resample(vel,[nzo,nxo],kind='cubic',ds=[4.0,8.0])

velsm = velsm.astype('float32')

velsmu = smooth(velsm,rect1=5,rect2=5)

plt.figure(2)
plt.imshow(velsmu,cmap='jet')
plt.show()

# Create data
prp = geom.defaultgeom(nxo,dx,nzo,dz,nsx=40,osx=4,dsx=10)

# Plot acquisition on model
#prp.plot_acq(velsmu)

ntu = 4000; dtu = 0.001; amp = 100
wav = bandpass(ntu,dtu,[2,10,12,20],amp,0.5)

plot_wavelet(wav,dtu,hbox=3)

# Forward modeling of data
dtd = 0.004
dat = prp.model_fulldata(velsmu,wav,dtd,dtu,verb=True)

viewimgframeskey(dat,transp=False,pclip=0.05)


