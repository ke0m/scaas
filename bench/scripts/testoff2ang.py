import inpout.seppy as seppy
import numpy as np
from scaas.adcig import convert2ang
from scaas.off2ang import off2ang
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
datw = datt[500:520,:,:]
#viewimgframeskey(dat.T,pclip=0.5,aratio=0.01)

# Window just one image point
iimg = datt[512,:,:]

#nta = 151; ota = -3; dta = 0.04
#nta = 301; ota = -3; dta = 0.02
nta = 601; ota = -3; dta = 0.01
nro = 1
#na = 101; oa = -70; da = 1.4
na = 281; oa = -70; da = 0.5

print(ota,ota+(nta-1)*dta)
print(oa,oa+(na-1)*da)
print(iimg.shape)

#iang = np.zeros([na,nt],dtype='float32')
angs = np.zeros([nx,na,nt],dtype='float32')
angw = np.zeros([20,na,nt],dtype='float32')
angp = np.zeros([20,na,nt],dtype='float32')

#plt.figure()
#plt.imshow(iimg,aspect='auto'); plt.show()
#convert2ang(1,nh,oh,dh,nta,ota,dta,na,oa,da,nt,ot,dt,4,iimg,iang,1,True)

#plt.imshow(iang.T,cmap='gray')
#plt.show()

# Parallel test
#print(datw.shape)
convert2ang(20,nh,oh,dh,nta,ota,dta,na,oa,da,nt,ot,dt,4,datw,angw,1,True)
#convert2ang(20,nh,oh,dh,nta,ota,dta,na,oa,da,nt,ot,dt,4,datw,angp,4,True)

# Shape, x,h,z
# Transp: h,x,z
# Want: h,z,x
datwt = np.ascontiguousarray(np.transpose(datw,(1,2,0)))
angwn = off2ang(datwt,oh,dh,dt,verb=True)

viewimgframeskey(angw,pclip=0.5,show=False)
viewimgframeskey(angwn,pclip=0.5,show=True)

# Full data
#convert2ang(nx,nh,oh,dh,nta,ota,dta,na,oa,da,nt,ot,dt,4,datt,angs,4,True)

#oaxes = seppy.axes([nt,na,nx],[ot,oa,ox],[dt,da,dx])
#sep.write_file(None,oaxes,angs.T,ofname='myangspar.H')

