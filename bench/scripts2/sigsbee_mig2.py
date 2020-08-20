import inpout.seppy as seppy
import numpy as np
import oway.coordgeom as geom
from utyls.signal import butter_bandpass_filter, ampspec1d
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in data
daxes,dat = sep.read_file("sigsbee_shotflatwind.H")
dat = np.ascontiguousarray(dat.reshape(daxes.n,order='F').T).astype('float32')
[nt,ntr] = daxes.n; [ot,_] = daxes.o; [dt,_] = daxes.d

# Read in velocity model
vaxes,vel = sep.read_file("sigsbee_velj4.H")
vel = vel.reshape(vaxes.n,order='F')
[nz,nx] = vaxes.n; [dz,dx] = vaxes.d; [oz,ox] = vaxes.o
ny = 1; dy = 1.0
velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel

# Read in coordinates
saxes,srcx = sep.read_file("sigsbee_srcxflatwind.H")
raxes,recx = sep.read_file("sigsbee_recxflatwind.H")
_,nrec = sep.read_file("sigsbee_nrecwind.H")
nrec = nrec.astype('int')

wei = geom.coordgeom(nx,dx,ny,dy,nz,dz,ox=ox,nrec=nrec,srcxs=srcx,recxs=recx)

waxes,wav = sep.read_file("sigwav.rsf",form='native')

img = wei.image_data(dat,dt,wav=wav,ntx=16,minf=1,maxf=51,jf=3,vel=velin,nrmax=20,
                     nthrds=24,wverb=False)

sep.write_file("mysigimgwind.H",img.T,ds=[dx,dy,dz],os=[ox,0.0,oz])

