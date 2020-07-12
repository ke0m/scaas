import numpy as np
import segyio
import inpout.seppy as seppy
import re
from oway.mute import mute
import matplotlib.pyplot as plt

def parse_text_header(segyfile):
  '''
  Format segy text header into a readable, clean dict
  '''
  raw_header = segyio.tools.wrap(segyfile.text[0])
  # Cut on C*int pattern
  cut_header = re.split(r'C ', raw_header)[1::]
  # Remove end of line return
  text_header = [x.replace('\n', ' ') for x in cut_header]
  text_header[-1] = text_header[-1][:-2]
  # Format in dict
  clean_header = {}
  i = 1
  for item in text_header:
    key = "C" + str(i).rjust(2, '0')
    i += 1
    clean_header[key] = item
  return clean_header

fdat = segyio.open("/scratch/data/sigsbee/ptest/sigsbee2a_nfs.sgy",ignore_geometry=True)
fvel = segyio.open("/scratch/data/sigsbee/ptest/sigsbee2a_migvel.sgy",ignore_geometry=True)

ft2km = 0.0003048

data = fdat.trace.raw[:]
[ntr,nt] = data.shape
dt = segyio.tools.dt(fdat)/1e6

velin  = fvel.trace.raw[:]*ft2km

j2 = 2; j1 = 4
vel = velin[::j2,::j1]

[nx,nz] = vel.shape
oz = 0;       dz = 0.00762*j1
ox = 3.05562; dx = 0.01143*j2

asrcx = np.asarray(fdat.attributes(segyio.TraceField.SourceX),dtype='float32')
arecx = np.asarray(fdat.attributes(segyio.TraceField.GroupX),dtype='float32')

# Convert to Km
asrcx *= (ft2km/100)
arecx *= (ft2km/100)

ebcdic_header = parse_text_header(fdat)

# Get total number of unique sources
srcx = np.unique(asrcx)
nsx = len(srcx)
dsx = srcx[1] - srcx[0]

# Get number of receivers per shot
nrecs = np.zeros(nsx,dtype='int')
for isx in range(nsx):
  sidxs = asrcx == srcx[isx]
  nrecs[isx] = len(arecx[sidxs])

# Get maximum number of receivers and receiver spacing
nrecmax = np.max(nrecs)
rec1 = arecx[sidxs]
drx = rec1[1] - rec1[0]
recx = np.zeros([nsx,nrecmax],dtype='float32')

# Output shot array
shots = np.zeros([nsx,nrecmax,nt],dtype='float32')

makefig = False
jrec = 20; zplt = 0.1
fsize = 16
for isx in range(nsx):
  # Get traces with this source index
  sidxs = asrcx == srcx[isx]
  recs  = arecx[sidxs]
  recss = recs[::jrec]
  isht = data[sidxs,:]
  # Outputs will be shots and receiver coordinates
  recx[isx,:nrecs[isx]] = recs[:]
  shots[isx,:nrecs[isx],:] = isht[:]
  # Plot data and acqusition
  if(makefig):
    fig,axar = plt.subplots(1,2,figsize=(15,10),gridspec_kw={'width_ratios':[3,1]})
    im = axar[0].imshow(vel.T,cmap='jet',interpolation='bilinear',extent=[ox,ox+(nx+8)*dx,nz*dz,oz])
    axar[0].scatter(srcx[isx],zplt,c="tab:red",marker='*',s=60)
    axar[0].set_xlabel('X (km)',fontsize=fsize)
    axar[0].set_ylabel('Z (km)',fontsize=fsize)
    axar[0].tick_params(labelsize=fsize)
    axar[0].scatter(recss,np.zeros(len(recss))+zplt,c='tab:green',marker='v')
    axar[1].imshow(shots[isx].T,cmap='gray',interpolation='sinc',vmin=-0.01,vmax=0.01,
                   extent=[0,nrecmax*drx,nt*dt,0],aspect=1.5)
    axar[1].set_xlabel('X (km)',fontsize=fsize)
    axar[1].set_ylabel('Time (s)',fontsize=fsize)
    axar[1].tick_params(labelsize=fsize)
    # Colorbar
    cbar_ax = fig.add_axes([0.6,0.37,0.01,0.25])
    cbar = fig.colorbar(im,cbar_ax,format='%.2f')
    cbar.ax.tick_params(labelsize=fsize)
    cbar.set_label(r'Velocity (km/s)',fontsize=fsize)
    plt.subplots_adjust(wspace=0.5)
    plt.savefig("./fig/subsamp/sigsbee%d.png"%(isx),bbox_inches='tight',transparent=True,dpi=150)
    plt.close()
    #plt.show()

muted = mute(shots,dt=dt,dx=0.075,v0=6.0,t0=1.0,half=False)

# Outputs
sep = seppy.sep()
sep.write_file("sigsbee_shots.H",muted.T,os=[0.0,0.0,0.0],ds=[dt,drx,dsx])
sep.write_file("sigsbee_recx.H",recx.T)
sep.write_file("sigsbee_srcx.H",srcx)

fdat.close(); fvel.close()

