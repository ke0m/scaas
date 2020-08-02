import inpout.seppy as seppy
import numpy as np
import scaas.defaultgeom as geom
from scaas.trismooth import smooth
import matplotlib.pyplot as plt
from utils.ptyprint import progressbar, create_inttag

sep = seppy.sep()

# Read in the model
vaxes,vel = sep.read_file("/home/joe/phd/projects/widb/Dat/marmvel.H")
vel = np.ascontiguousarray(vel.reshape(vaxes.n,order='F')).astype('float32')
[nz,nx] = vaxes.n; [oz,ox] = vaxes.o; [dz,dx] = vaxes.d

# Resample the model
j1 = 3; j2 = 3
vel1   = smooth(vel,rect1=2,rect2=2)
velr   = vel1[::j2,::j1]
velrsm = smooth(velr,rect1=50,rect2=50)

[nzr,nxr] = velr.shape; dzr = dz*j2; dxr = dx*j1

nsx = 2; osx=int(nxr/10); dsx = 1; bx = 50; bz = 50
prp = geom.defaultgeom(nxr,dxr,nzr,dzr,nsx=nsx,osx=osx,dsx=dsx,bx=bx,bz=bz)

[nzr,nxr] = velr.shape; dzr = dz*j2; dxr = dx*j1

# Read in the movies
faxes,fmov = sep.read_file("fmov.H")
fmov = fmov.reshape(faxes.n,order='F').T

saxes,smov = sep.read_file("smov.H")
smov = smov.reshape(saxes.n,order='F').T

laxes,lmov = sep.read_file("lmov.H")
lmov = lmov.reshape(laxes.n,order='F').T

gaxes,gmov = sep.read_file("gmov.H")
gmov = gmov.reshape(gaxes.n,order='F').T

nfr = gmov.shape[0]

# Get source and receiver coordinates
srcx = prp.allsrcx[0]; srcz = prp.allsrcz[0]
recx = prp.allrecx[0]; recz = prp.allrecz[0]
recx = recx[::12]; zeros = np.zeros(len(recx))

dzrk = dzr*0.001; dxrk = dxr*0.001

# # Make full wavefield movie
fsize = 16
for ifr in progressbar(range(nfr),"ifr"):
  fig = plt.figure(figsize=(14,4)); ax = fig.gca()
  im = ax.imshow(velr/1000.0,cmap='jet',extent=[0,nxr*dxrk,nzr*dzrk,0],interpolation='bilinear')
  ax.imshow(fmov[ifr],cmap='RdGy',extent=[0,nxr*dxrk,nzr*dzrk,0],vmin=-2,vmax=2,alpha=0.6,interpolation='bilinear')
  ax.scatter(recx*dxrk-50*dxrk,zeros + 3*dzrk,marker='v',c='tab:green')
  ax.scatter(srcx[0]*dxrk-55*dxrk,5*dzrk,marker='*',c='tab:red',s=100)
  ax.set_xlabel('X (km)',fontsize=fsize)
  ax.set_ylabel('Z (km)',fontsize=fsize)
  ax.tick_params(labelsize=fsize)
  cbar_ax = fig.add_axes([0.86,0.13,0.02,0.73])
  cbar = fig.colorbar(im,cbar_ax,format='%.2f')
  cbar.ax.tick_params(labelsize=fsize)
  cbar.solids.set(alpha=0.3)
  cbar.set_label('Sound speed (km/s)',fontsize=fsize)
  fnum = create_inttag(ifr,nfr)
  plt.savefig('./fig/rehman/fwd%s.png'%(fnum),bbox_inches='tight',transparent=True,dpi=150)
  plt.close()
  plt.show()

# Gradient movie
#fsize = 12
# for ifr in progressbar(range(nfr),"ifr"):
#   fig,axarr = plt.subplots(1,3,figsize=(14,6))
#   axarr[0].imshow(smov[ifr],cmap='gray',extent=[0,nxr*dxrk,nzr*dzrk,0],interpolation='bilinear',vmin=-5e4,vmax=5e4)
#   axarr[0].set_xlabel('X (km)',fontsize=fsize)
#   axarr[0].set_ylabel('Z (km)',fontsize=fsize)
#   axarr[0].tick_params(labelsize=fsize)
#   axarr[1].imshow(lmov[ifr],cmap='gray',extent=[0,nxr*dxrk,nzr*dzrk,0],interpolation='bilinear',vmin=-1,vmax=1)
#   axarr[1].set_xlabel('X (km)',fontsize=fsize)
#   axarr[1].tick_params(labelsize=fsize)
#   #axarr[2].imshow(velr/1000.0,cmap='jet',extent=[0,nxr*dxrk,nzr*dzrk,0],interpolation='bilinear')
#   axarr[2].imshow(gmov[ifr],cmap='gray',extent=[0,nxr*dxrk,nzr*dzrk,0],interpolation='bilinear',vmin=-2e5,vmax=2e5,
#                   alpha=1.0)
#   axarr[2].set_xlabel('X (km)',fontsize=fsize)
#   axarr[2].tick_params(labelsize=fsize)
#   fnum = create_inttag(ifr,nfr)
#   plt.savefig('./fig/rehman/img/img%s.png'%(fnum),bbox_inches='tight',transparent=True,dpi=150)
#   plt.close()
    #plt.show()
