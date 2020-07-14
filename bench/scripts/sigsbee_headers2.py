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

# Get number of receivers per shot
nrecs = np.zeros(nsx,dtype='float32')
for isx in range(nsx):
  sidxs = asrcx == srcx[isx]
  nrecs[isx] = len(arecx[sidxs])

# Output shot array
shots = []
recx = []

for isx in range(nsx):
  # Get traces with this source index
  sidxs = asrcx == srcx[isx]
  recs  = arecx[sidxs]
  recx.append(recs)
  isht  = data[sidxs,:]
  mut   = mute(isht,dt=dt,dx=0.075,v0=6.0,t0=1.0,half=False)
  shots.append(np.squeeze(mut))

muted = np.concatenate(shots,axis=0)
recxs = np.concatenate(recx,axis=0)

plt.show()
plt.imshow(muted[:1000,:].T,cmap='gray',interpolation='sinc',vmin=-0.01,vmax=0.01)
plt.show()

# Outputs
sep = seppy.sep()
sep.write_file("sigsbee_shotflat.H",muted.T,ds=[dt,1.0])
sep.write_file("sigsbee_recxflat.H",recxs)
sep.write_file("sigsbee_srcxflat.H",srcx)
sep.write_file("sigsbee_nrec.H",nrecs)
#sep.write_file("sigsbee_vel.H",vel.T,os=[oz,ox],ds=[dz,dx])

fdat.close(); fvel.close()

