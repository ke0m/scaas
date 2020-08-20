import numpy as np
import scaas.defaultgeom as geom
from scaas.wavelet import ricker
from resfoc.resmig import preresmig, get_rho_axis
from utyls.movie import viewimgframeskey
from utyls.plot import plot_wavelet, plot_imgpoff, plot_imgpang
from utyls.signal import ampspec2d
import matplotlib.pyplot as plt

# Create a single point scatterer model
nx = 101; dx = 10
nz = 101; dz = 10
mod = np.zeros([nz,nx],dtype='float32')
mod[int(nz/2),int(nz/2)] = 1.0

# Create background wavespeed
vel = np.zeros(mod.shape,dtype='float32') + 2000.0

# Create modeling class
prp = geom.defaultgeom(nx,dx,nz,dz,nsx=21,dsx=5,bx=50,bz=50)
#prp.plot_acq(mod,cmap='gray',vmin=-1,vmax=1)

# Create wavelet
ntu = 2000; dtu = 0.001; amp = 100
wav = ricker(ntu,dtu,20,100,0.5)
#plot_wavelet(wav,dtu)

dtd = 0.004
dat = prp.model_lindata(vel,mod,wav,dtd,dtu,nthrds=21,verb=True)

#viewimgframeskey(dat,transp=False)

# Image the data
img = prp.wem(vel,dat,wav,dtd,dtu,nthrds=21,verb=True)

#plt.figure(1)
plt.imshow(img,cmap='gray',extent=[0,(nx-1)*dx/1000,(nz-1)*dz/1000,0],interpolation='sinc')
spc,kz,kx = ampspec2d(img,dz/1000,dx/1000)
plt.figure(2)
plt.imshow(spc,cmap='jet',extent=[kx[0],kx[-1],kz[0],kz[-1]],interpolation='bilinear')
plt.show()

## Perform extended imaging
eimg = prp.wem(vel,dat,wav,dtd,dtu,nthrds=21,verb=True,nh=10)
nh,oh,dh = prp.get_off_axis()
#plot_imgpoff(eimg,dx/1000.0,dz/1000.0,zoff=int(nh/2),xloc=int((nx-1)/2),oh=oh/1000.0,dh=dh/1000.0,hmax=0.1)
viewimgframeskey(eimg,transp=False,interp='sinc')

# Residual migration
#eimgt = np.ascontiguousarray(np.transpose(eimg,(0,2,1)))
#TODO: put transpose option in preresmig
storm = preresmig(eimg,[dh,dz,dx],nro=10,transp=True)
nro,oro,dro = get_rho_axis(nro=10,oro=0.0,dro=0.01)

#viewimgframeskey(storm[:,int(nh/2),:,:],interp='sinc',ttlstring="rho=%f",ottl=0.91,dttl=0.01)
#viewimgframeskey(storm[:,int(nh/2),:,:],transp=False,interp='sinc',ttlstring="rho=%.2f",ottl=0.91,dttl=0.01)

# Convert to angle
#aimg = prp.to_angle(eimg,transp=False,verb=True)
storma = prp.to_angle(storm,transp=False,nthrds=21,verb=True)
na,oa,da = prp.get_ang_axis()

#aimgt = np.transpose(aimg,(1,2,0))
stormat1 = np.transpose(storma[int((nro)/2)+5],(1,2,0))
stormat2 = np.transpose(storma[int((nro)/2)+0],(1,2,0))
stormat3 = np.transpose(storma[int((nro)/2)-5],(1,2,0))
plot_imgpang(stormat1,dx/1000.0,dz/1000.0,int((nx-1)/2),oa,da,show=False)
plot_imgpang(stormat2,dx/1000.0,dz/1000.0,int((nx-1)/2),oa,da,show=False)
plot_imgpang(stormat3,dx/1000.0,dz/1000.0,int((nx-1)/2),oa,da,show=True)

#viewimgframeskey(aimgt,interp='sinc')


