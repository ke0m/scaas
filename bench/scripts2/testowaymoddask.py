import inpout.seppy as seppy
import numpy as np
from scaas.wavelet import ricker
import oway.defaultgeomnode as geom
from  scaas.trismooth import smooth
import matplotlib.pyplot as plt
from utils.plot import plot_wavelet
from utils.movie import viewimgframeskey
from dask.distributed import Client, SSHCluster, progress

# Create Dask cluster
cluster = SSHCluster(
                     ["localhost", "fantastic", "thing"],
                     connect_options={"known_hosts": None},
                     worker_options={"nthreads": 1},
                     scheduler_options={"port": 0, "dashboard_address": ":8797"}
                    )

client = Client(cluster)


sep = seppy.sep()

# Dimensions
nx = 800; dx = 0.015
ny = 1;    dy = 0.125
nz = 400;  dz = 0.005

# Build input slowness
vz = 1.5 +  np.linspace(0.0,dz*(nz-1),nz)
vel = np.ascontiguousarray(np.repeat(vz[np.newaxis,:],nx,axis=0).T)

velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel[:]

# Build reflectivity
ref = np.zeros(velin.shape,dtype='float32')
ref[349,0,49:749] = 1.0
npts = 25
refsm = smooth(smooth(smooth(ref,rect1=npts),rect1=npts),rect1=npts)

# Create ricker wavelet
n1 = 2000; d1 = 0.004;
freq = 8; amp = 0.5; dly = 0.2;
wav = ricker(n1,d1,freq,amp,dly)

osx = 50; dsx = 50; nsx = 10
wei = geom.defaultgeomnode(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                           nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

# Create the frequency domain source
wfft = wei.fft1(wav,d1,minf=1.0,maxf=31.0)

dchunks = wei.create_mod_chunks(nsx,wfft)
wei.set_model_pars(velin,ref,ntx=15,px=112,nthrds=10,wverb=False)

futures = []; bs = []
# First scatter
bs = client.scatter(dchunks)

# Now run
for ib in bs:
  x = client.submit(wei.model_chunk,ib)
  futures.append(x)

result = [future.result() for future in futures]

nw,ow,dw = wei.get_freq_axis()

for ires in result:
  plt.imshow(np.real(ires).T,cmap='gray',interpolation='sinc')
  plt.show()

#sep.write_file("mycmplxdat.H",chnk1['dat'].T,os=[0,0,ow,0,0],ds=[dx,dy,dw,1.0,1.0])
#sep.write_file("mydat.H",dat.T,os=[0,0,0,osx,0],ds=[d1,dx,dy,dsx,1.0])

#plt.figure(figsize=(10,10))
#plt.imshow(np.real(dat[0,0,:,0,:]),cmap='gray',interpolation='sinc')
#plt.figure(figsize=(10,10))
#plt.imshow(np.real(dat[0,1,:,0,:]),cmap='gray',interpolation='sinc')
#plt.show()

