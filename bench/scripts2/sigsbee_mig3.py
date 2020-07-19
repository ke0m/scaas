import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in data
daxes,dat = sep.read_file("sigsbee_shotflat2.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

# Read in velocity model
vaxes,vel = sep.read_file("sigsbee_velj1.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nvx] = vaxes.n; [dz,dvx] = vaxes.d; [oz,ovx] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nvx],dtype='float32')
velin[:,0,:] = vel

# Read in coordinates
saxes,srcx = sep.read_file("sigsbee_srcxflat2.H")
raxes,recx = sep.read_file("sigsbee_recxflat2.H")
_,nrec = sep.read_file("sigsbee_nrec2.H")
nrec = nrec.astype('int')

# Imaging grid (same as shots)
nxi = saxes.n[0]; oxi = srcx[0]; dxi = srcx[1] - srcx[0]
print("Image grid: nxi=%d oxi=%f dxi=%f"%(nxi,oxi,dxi))

wei = geom.coordgeom(nxi,dxi,ny,dy,nz,dz,ox=oxi,nrec=nrec,srcxs=srcx,recxs=recx)

velint = wei.interp_vel(velin,dvx,dy,ovx=ovx)

img = wei.image_data(dat,dt,ntx=16,minf=1,maxf=51,vel=velint,nrmax=20,nthrds=24,wverb=False)

sep.write_file("mysigimg2.H",img,ds=[dz,dy,dxi],os=[oz,0.0,oxi])

