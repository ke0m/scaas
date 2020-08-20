import inpout.seppy as seppy
import numpy as np
import zmq
from scaas.wavelet import ricker
from oway.modelchunkr import modelchunkr
from oway.coordgeomchunk import default_coord, coordgeomchunk
from scaas.trismooth import smooth
from server.distribute import dstr_collect, dstr_sum
from client.sshworkers import launch_sshworkers, kill_sshworkers
from utyls.plot import plot_wavelet
from utyls.movie import viewimgframeskey
import matplotlib.pyplot as plt

sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','vision','thing']
cfile = "/homes/sep/joseph29/projects/scaas/oway/modelsshworker.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1)

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

# Create coordinates from default geometry
osx = 20; dsx = 10; nsx = 70
srcx,srcy,nrec,recx,recy = default_coord(nx,dx,ny,dy,nz,dz,
                                         nsx=nsx,dsx=dsx,osx=osx,
                                         nsy=1,dsy=1.0)

# Create generator
nchnk = len(hosts)
miter = modelchunkr(nchnk,
                    dx,dy,dz,refsm,velin,wav,d1,minf=1.0,maxf=31.0,
                    nrec=nrec,srcx=srcx,srcy=srcy,recx=recx,recy=recy)
miter.set_model_pars(ntx=15,px=112,nthrds=40,sverb=True)
gen = iter(miter)

# Bind to socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

# Distribute work to workers
okeys = ['result','cid']
output = dstr_collect(okeys,nchnk,gen,socket)

odat = miter.reconstruct_data(output,dly,reg=True)

plt.figure()
plt.imshow(odat[0].T,cmap='gray',interpolation='sinc')
plt.figure()
plt.imshow(odat[-1].T,cmap='gray',interpolation='sinc')
plt.show()

sep.write_file("mydatnode2.H",odat.T,os=[0,0,osx],ds=[d1,dx,dsx])

kill_sshworkers(cfile,hosts,verb=False)


