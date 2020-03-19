import numpy as np
import inpout.seppy as seppy
from scaas.trismooth import smooth
from utils.movie import viewimgframeskey
import matplotlib.pyplot as plt

n = 400
arr1d = np.zeros(n,dtype='float32')

arr1d[int(n/2)] = 1

#sm1d = smooth(arr1d,rect1=3)

#plt.plot(sm1d); plt.show()

#arr2d = np.zeros([n,2*n],dtype='float32')
#arr2d[0,int(n/2)-1] = 1
#sm2d = smooth(arr2d,rect1=3,rect2=3)

#plt.figure(1)
#plt.imshow(sm2d.T,cmap='gray',vmin=-0.1,vmax=0.1);

sep = seppy.sep([])

#daxes,dat = sep.read_file(None,ifname='/home/joe/phd/projects/scaas/scaas/src/spike2dsm.H')

#dat = dat.reshape(daxes.n,order='F')

#plt.figure(2)
#plt.imshow(dat,cmap='gray',vmin=-0.1,vmax=0.1)
#plt.show()


#arr3d = np.zeros([n,n,n],dtype='float32')
#arr3d[int(n/2),-3,int(n/2)] = 1
#sm3d = smooth(arr3d,rect1=3,rect2=3,rect3=3)

#viewimgframeskey(sm3d,vmin=-0.1,vmax=0.1)

# Read in velocity model
vaxes,vel = sep.read_file(None,ifname='/home/joe/phd/projects/fwitut/Dat/marmvel.H')
saxes,smo = sep.read_file(None,ifname='marmsm.H')

vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F')).astype('float32')
smo = np.ascontiguousarray(smo.reshape(vaxes.n,order='F')).astype('float32')

plt.figure(4)
plt.imshow(vel,cmap='jet')

velsm = smooth(vel,rect1=150,rect2=150)

plt.figure(5)
plt.imshow(velsm,cmap='jet')

plt.figure(6)
plt.imshow(smo,cmap='jet')

plt.show()

