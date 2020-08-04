import inpout.seppy as seppy
import numpy as np
import oway.defaultgeomnode as geom
import matplotlib.pyplot as plt
from utils.movie import viewimgframeskey
from dask.distributed import Client, SSHCluster, progress

# Read in the data
sep = seppy.sep()

daxes,dat = sep.read_file("mydatnode.H")
dat = dat.reshape(daxes.n,order='F')
[nt,nx,nsx] = daxes.n; [dt,dx,dsx] = daxes.d; [ot,ox,osx] = daxes.o

# Prepare the data
datt = dat.T
datin = np.ascontiguousarray(datt.reshape([1,nsx,1,nx,nt]))

# Create dask cluster
cluster = SSHCluster(
                     ["localhost", "fantastic", "thing"],
                     connect_options={"known_hosts": None},
                     worker_options={"nthreads": 1, "nprocs": 1, "memory_limit": 20e9},
                     scheduler_options={"port": 0, "dashboard_address": ":8797"}
                    )

client = Client(cluster)

# Dimensions
nx = 800; dx = 0.015
ny = 1;   dy = 0.125
nz = 400; dz = 0.005

# Build input slowness
vz = 1.5 +  np.linspace(0.0,dz*(nz-1),nz)
vel = np.ascontiguousarray(np.repeat(vz[np.newaxis,:],nx,axis=0).T)

velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel[:]

wei = geom.defaultgeomnode(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                           nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

img = wei.image_data(datin,dt,minf=1.0,maxf=31.0,vel=velin,nhx=20,ntx=15,
                     nthrds=30,client=client)

viewimgframeskey(img[0,:,:,0],transp=False,cmap='gray')

#plt.imshow(img[:,0,:],cmap='gray')
#plt.show()

