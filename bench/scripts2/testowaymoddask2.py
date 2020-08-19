import inpout.seppy as seppy
import numpy as np
from scaas.wavelet import ricker
import oway.defaultgeomnode as geom
from  scaas.trismooth import smooth
import matplotlib.pyplot as plt
from utils.plot import plot_wavelet
from utils.movie import viewimgframeskey
from dask.distributed import Client, SSHCluster, progress
from cluster.daskutils import shutdown_sshcluster

sep = seppy.sep()

# Dimensions
nx = 800; dx = 0.015
ny = 1;   dy = 0.125
nz = 400; dz = 0.005

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
freq = 8; amp = 0.5; dly = 0.2; it0 = int(dly/d1)
wav = ricker(n1,d1,freq,amp,dly)

osx = 20; dsx = 10; nsx = 70
wei = geom.defaultgeomnode(nx=nx,dx=dx,ny=ny,dy=dy,nz=nz,dz=dz,
                           nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)

wei.plot_acq(velin)

# Create Dask cluster
hosts = ["localhost", "torch", "thing", "jarvis"]
cluster = SSHCluster(
                     hosts,
                     connect_options={"known_hosts": None},
                     worker_options={"nthreads": 1, "nprocs": 1, "memory_limit": 20e9, "worker_port": '33149:33150'},
                     scheduler_options={"port": 0, "dashboard_address": ":8797"}
                    )

client = Client(cluster)

odatr = wei.model_data(wav,d1,dly,minf=1.0,maxf=31.0,vel=velin,ref=refsm,ntx=15,px=112,
                       nthrds=40,client=client)

plt.figure()
plt.imshow(odatr[0].T,cmap='gray',interpolation='sinc')
plt.figure()
plt.imshow(odatr[-1].T,cmap='gray',interpolation='sinc')
plt.show()

sep.write_file("mydatnode.H",odatr.T,os=[0,0,osx],ds=[d1,dx,dsx])

# Shutdown dask
client.shutdown()
shutdown_sshcluster(hosts)

