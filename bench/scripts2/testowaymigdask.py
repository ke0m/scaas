import inpout.seppy as seppy
import numpy as np
import oway.defaultgeomnode as geom
import matplotlib.pyplot as plt
from utyls.movie import viewimgframeskey
from dask.distributed import Client, SSHCluster, progress

sep = seppy.sep()

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
                     worker_options={"nthreads": 1},
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

# Reshape the data
datinr = datin.reshape([nsx*nx,nt])

# FFT the data
datw = wei.fft1(datinr,dt,minf=1.0,maxf=31.0)

# FFT the wavelet
wav  = np.zeros(nt,dtype='float32')
wav[0] = 1.0
wavw = wei.fft1(wav,dt,minf=1.0,maxf=31.0)

# Chunk the data
nchnks = len(client.cluster.workers)
dchunks = wei.create_img_chunks(nchnks,wavw,datw)

# Set the imaging pars
wei.set_image_pars(velin,nhx=20,ntx=15,nthrds=30,wverb=False)

futures = []; bs = []
# First scatter
bs = client.scatter(dchunks)

# Now run
for ib in bs:
  x = client.submit(wei.image_chunk,ib)
  futures.append(x)

progress(futures)
result = [future.result() for future in futures]

# Sum all chunks
imgs = np.sum(np.asarray(result),axis=0)

viewimgframeskey(imgs[0,:,:,0],transp=False,cmap='gray')

#plt.imshow(imgs[:,0,:],cmap='gray')
#plt.show()

