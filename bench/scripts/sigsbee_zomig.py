import inpout.seppy as seppy
import numpy as np
#import oway.zerooffset_kiss as zo
import oway.zerooffset as zo
import matplotlib.pyplot as plt
import time

sep = seppy.sep()

# Read in data
daxes,dat = sep.read_file("sigsbee_zodata.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
dat = np.expand_dims(dat,axis=0)
[nt,nxi] = daxes.n; [ot,oxi] = daxes.o; [dt,dxi] = daxes.d

# Read in velocity model
vaxes,vel = sep.read_file("sigvel.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

zoimg = zo.zerooffset(nxi,dxi,ny,dy,nz,dz,ox=oxi)

velint = zoimg.interp_vel(velin,dvx,dy,ovx=ovx)

beg = time.time()
img = zoimg.image_data(dat,dt,ntx=4,minf=0,maxf=60.1,jf=1,vel=velint,nrmax=3,
                     nthrds=4,wverb=True)
print("Elapsed=%f"%(time.time()-beg))

sep.write_file("myzosigimg.H",img.T,ds=[dxi,dy,dz],os=[oxi,0.0,oz])

