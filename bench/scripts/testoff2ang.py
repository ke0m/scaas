import inpout.seppy as seppy
import numpy as np
from scaas.adcig import convert2ang
from utils.movie import viewimgframeskey
import matplotlib.pyplot as plt

sep = seppy.sep([])

# Read in the offset gathers
daxes,dat = sep.read_file(None,ifname='fltprepangrsf.H')

dat = dat.reshape(daxes.n,order='F')

nt = daxes.n[0]; ot = daxes.o[0]; dt = daxes.d[0]
nh = daxes.n[1]; oh = daxes.o[1]; dh = daxes.d[1]
nx = daxes.n[2]; ox = daxes.o[2]; dx = daxes.d[2]

datt = np.ascontiguousarray(dat.T).astype('float32')
#viewimgframeskey(dat.T,pclip=0.5,aratio=0.01)

# Window just one image point
iimg = datt[512,:,:]

nta = 151; ota = -3; dta = 0.04
#nta = 301; ota = -3; dta = 0.02
#nta = 601; ota = -3; dta = 0.01
nro = 1
na = 101; oa = -70; da = 1.4

print(ota,ota+(nta-1)*dta)
print(oa,oa+(na-1)*da)

#iang = np.zeros([na,nt],dtype='float32')
angs = np.zeros([nx,na,nt],dtype='float32')

#plt.figure()
#plt.imshow(iimg,aspect='auto'); plt.show()
#convert2ang(1,1,nh,oh,dh,nta,ota,dta,na,oa,da,nt,ot,dt,4,iimg,iang)

#plt.imshow(iang.T,cmap='gray')
#plt.show()

convert2ang(1,nx,nh,oh,dh,nta,ota,dta,na,oa,da,nt,ot,dt,4,datt,angs)

viewimgframeskey(angs,pclip=0.5)

