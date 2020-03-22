import inpout.seppy as seppy
import numpy as np
import scaas.slantstk as slantstk
import matplotlib.pyplot as plt

sep = seppy.sep([])

daxes,dat = sep.read_file(None,ifname='angspk.H')
taxes,tan = sep.read_file(None,ifname='tanspk.H')

nt = daxes.n[0]; ot = daxes.o[0]; dt = daxes.d[0]
nx = daxes.n[1]; ox = daxes.o[1]; dx = daxes.d[1]
ns = taxes.n[1]; os = taxes.o[1]; ds = taxes.d[1]

dat = dat.reshape(daxes.n,order='F')
tan = tan.reshape(taxes.n,order='F')

plt.figure(1)
plt.imshow(dat,cmap='gray',aspect=0.1,vmin=-1,vmax=1)
plt.figure(2)
plt.imshow(tan,cmap='gray',aspect=5.0,vmin=-1,vmax=1)

ssk = slantstk.slantstk(True,nx,ox,dx,ns,os,ds,nt,ot,dt,0,1)

mytan = np.zeros([ns,nt],dtype='float32')
myang = np.zeros([nx,nt],dtype='float32')
myang[:] = dat[:].T

ssk.adjoint(False,nt*ns,nt*nx,mytan,myang)

plt.figure(3)
plt.imshow(mytan.T,cmap='gray',aspect=5.0,vmin=-1,vmax=1)

# Read in a real offset gather
oaxes,off = sep.read_file(None,ifname='oneang.H')
raxes,ran = sep.read_file(None,ifname='onetan.H')

off = off.reshape(oaxes.n,order='F')
ran = ran.reshape(raxes.n,order='F')

myoff = np.zeros([nx,nt],dtype='float32')
myran = np.zeros([ns,nt],dtype='float32')
myoff[:] = off[:].T

ssk.adjoint(False,nt*ns,nt*nx,myran,myoff)

plt.figure(4)
plt.imshow(off,cmap='gray',aspect=0.1)

plt.figure(5)
plt.imshow(ran,cmap='gray',aspect=5.0)

plt.figure(6)
plt.imshow(myran.T,cmap='gray',aspect=5.0)

plt.show()

