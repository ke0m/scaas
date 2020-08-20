import inpout.seppy as seppy
import numpy as np
import zmq
from scaas.wavelet import ricker
from oway.imagechunkr import imagechunkr
from oway.coordgeomchunk import default_coord, coordgeomchunk
from scaas.trismooth import smooth
from server.distribute import dstr_collect, dstr_sum
from client.sshworkers import launch_sshworkers, kill_sshworkers
from utyls.plot import plot_wavelet
from utyls.movie import viewimgframeskey
import matplotlib.pyplot as plt

sep = seppy.sep()

# Read in the data
sep = seppy.sep()

# Start workers
hosts = ['fantastic','storm','thing','vision']
cfile = "/homes/sep/joseph29/projects/scaas/oway/imagesshworker.py"
launch_sshworkers(cfile,hosts=hosts,sleep=1,verb=1)

daxes,dat = sep.read_file("mydatnode2.H")
dat = dat.reshape(daxes.n,order='F')
[nt,nx,nsx] = daxes.n; [dt,dx,dsx] = daxes.d; [ot,ox,osx] = daxes.o

# Prepare the data
datt = dat.T
datin = np.ascontiguousarray(datt.reshape([nsx*nx,nt]))

# Dimensions
nx = 800; dx = 0.015
ny = 1;   dy = 0.125
nz = 400; dz = 0.005

# Build input slowness
vz = 1.5 +  np.linspace(0.0,dz*(nz-1),nz)
vel = np.ascontiguousarray(np.repeat(vz[np.newaxis,:],nx,axis=0).T)

velin = np.zeros([nz,ny,nx],dtype='float32')
velin[:,0,:] = vel[:]

osx = 20; dsx = 10; nsx = 70
srcx,srcy,nrec,recx,recy = default_coord(nx,dx,ny,dy,nz,dz,
                                         nsx=nsx,dsx=dsx,osx=osx,nsy=1,dsy=1.0)
nchnk = len(hosts)
icnkr = imagechunkr(nchnk,
                    nx,dx,ny,dy,nz,dz,velin,datin,dt,minf=1.0,maxf=31.0,
                    nrec=nrec,srcx=srcx,srcy=srcy,recx=recx,recy=recy)

icnkr.set_image_pars(ntx=15,nhx=20,nthrds=40,sverb=True)
gen = iter(icnkr)

# Bind to socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://0.0.0.0:5555")

# Distribute work to workers and sum over results
oimg = dstr_sum('cid','result',nchnk,gen,socket,icnkr.get_img_shape())

#print(oimg.shape)
viewimgframeskey(oimg[0,:,:,0,:])

kill_sshworkers(cfile,hosts,verb=False)

