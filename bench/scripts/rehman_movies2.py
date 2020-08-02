import inpout.seppy as seppy
import numpy as np
import scaas.defaultgeom as geom
import scaas.scaas2dpy as sca2d
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
velrsm = smooth(velr,rect1=50,rect2=50)

[nzr,nxr] = velr.shape; dzr = dz*j2; dxr = dx*j1

nsx = 2; osx=int(nxr/10); dsx = 1; bx = 50; bz = 50
prp = geom.defaultgeom(nxr,dxr,nzr,dzr,nsx=nsx,osx=osx,dsx=dsx,bx=bx,bz=bz)

prp.build_taper(60,140)

# Make the wavelet
ntu = 4000; dtu = 0.001
freq = 20; amp = 100.0; dly=0.2
wav = ricker(ntu,dtu,freq,amp,dly)

# Pad the velocity model
velrp   = prp.pad_model(velr  ,tvel=1500.0,zcut=0 )
velrsmp = prp.pad_model(velrsm,tvel=1500.0,zcut=53)

[nzp,nxp] = velrp.shape
bx = 50; bz = 50; alpha = 0.99

sca = sca2d.scaas2d(ntu,nxp,nzp,dtu,dxr,dzr,dtu,bx,bz,alpha)

srcx = prp.allsrcx[0]; srcz = prp.allsrcz[0]
nsrc = len(srcx)
recx = prp.allrecx[0]; recz = prp.allrecz[0]
nrec = len(recx)
dattru = np.zeros([ntu,nrec],dtype='float32')
datmod = np.zeros([ntu,nrec],dtype='float32')

# Forward wavefield
fwfld = np.zeros([ntu,nzp,nxp],dtype='float32')
swfld = np.zeros([ntu,nzp,nxp],dtype='float32')

# Full wavefield
jw = 2
sca.fwdprop_wfld(wav,srcx,srcz,nsrc,velrp,fwfld)
fwfldsmp = np.copy(fwfld[::jw,...])
del fwfld

# Second time derivative of background
sca.fwdprop_wfld(wav,srcx,srcz,nsrc,velrsmp,swfld)
s2t = np.zeros(swfld.shape,dtype='float32')
sca.d2t(swfld,s2t)
del swfld
s2tsmp = np.copy(s2t[::jw,...])
del s2t

sca.fwdprop_oneshot(wav,srcx,srcz,nsrc,recx,recz,nrec,velrp  ,dattru)
sca.fwdprop_oneshot(wav,srcx,srcz,nsrc,recx,recz,nrec,velrsmp,datmod)

fsize = 12
fig = plt.figure(figsize=(5,15)); ax = fig.gca()
ax.imshow(dattru,cmap='gray',interpolation='sinc',vmin=-2,vmax=2,extent=[0,nxr*dxr/1000.0,ntu*dtu,0],aspect=1.0)
ax.set_xlabel('X (km)',fontsize=fsize)
ax.set_ylabel('Time (s)',fontsize=fsize)
ax.tick_params(labelsize=fsize)
#plt.show()
plt.savefig('./fig/rehman/dat.png',bbox_inches='tight',dpi=150,transparent=True)

#asrc = (datmod - dattru)
#
#lsol = np.zeros([ntu,nzp,nxp],dtype='float32')
#sca.adjprop_wfld(asrc,recx,recz,nrec,velrsmp,lsol)
#lsolsmp = np.copy(lsol[::jw,...])
#del lsol
#
#jm = 5
#gmovie = np.cumsum(lsolsmp*s2tsmp,axis=0)
#gmovsmp = gmovie[::jm,...]
#
#nf = gmovsmp.shape[0]
#
#for ifr in range(nf):
#  gmovsmp[ifr] *= prp.tap
#
#gmovfin = prp.trunc_model(gmovsmp)
#lmovfin = prp.trunc_model(lsolsmp)
#fmovfin = prp.trunc_model(fwfldsmp)
#smovfin = prp.trunc_model(s2tsmp)
#
## Write these movies to files
#sep.write_file("fmov.H",fmovfin[::jm,...].T,ds=[dxr,dzr,1.0])
#sep.write_file("smov.H",smovfin[::jm,...].T,ds=[dxr,dzr,1.0])
#sep.write_file("lmov.H",lmovfin[::jm,...].T,ds=[dxr,dzr,1.0])
#sep.write_file("gmov.H",gmovfin.T,ds=[dxr,dzr,1.0])

#viewimgframeskey(fmovfin[::jm,...],transp=False,cmap='seismic',hbox=5,wbox=14,interp='bilinear',vmin=-1,vmax=1)
#viewimgframeskey(smovfin[::jm,...],transp=False,cmap='seismic',hbox=5,wbox=14,interp='bilinear',vmin=-1e5,vmax=1e5)
#viewimgframeskey(lmovfin[::jm,...],transp=False,cmap='seismic',hbox=5,wbox=14,interp='bilinear',vmin=-2,vmax=2)
#viewimgframeskey(gmovfin,transp=False,cmap='seismic',vmin=-3e5,vmax=3e5,hbox=5,wbox=14,interp='bilinear')
