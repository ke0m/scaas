import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
from utils.signal import butter_bandpass_filter, ampspec1d
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in data
daxes,dat = sep.read_file("sigsbee_shots.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,nrecmax,nexp] = daxes.n; [ot,orx,oexp] = daxes.o; [dt,drx,dexp] = daxes.d
dat = dat.reshape([nexp,1,nrecmax,nt])

# Read in velocity model
vaxes,vel = sep.read_file("sigsbee_vel.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nx] = vaxes.n; [dz,dx] = vaxes.d; [oz,ox] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel

# Read in coordinates
saxes,srcx = sep.read_file("sigsbee_srcx.H")
isrcx = (srcx - ox)/dx + 0.5
isrcx = np.abs(isrcx).astype('int')

raxes,recx = sep.read_file("sigsbee_recx.H")
recx = np.ascontiguousarray(recx.reshape(raxes.n,order='F').T)
irecx = (recx - ox)/dx + 0.5
ridx = irecx < 0
irecx[ridx] = 0
irecx = irecx.astype('int')

wei = geom.coordgeom(nx,dx,ny,dy,nz,dz,ox=ox,srcxs=isrcx,recxs=irecx)

waxes,hwav = sep.read_file("../oway/src/srmodmig/sigwav.rsf",form='native')

# Make wavelet
spk = np.zeros(nt,dtype='float32')
spk[0] = 1
wav = butter_bandpass_filter(spk,locut=12,hicut=28,fs=1/dt,order=1,zrophz=False)
wfft,fs = ampspec1d(wav,dt)
hfft,fs = ampspec1d(hwav,dt)

wei.image_data(dat,dt,wav=hwav,ntx=16,minf=1,maxf=51,jf=3,vel=velin,nrmax=20)


