import numpy as np
import segyio
import inpout.seppy as seppy
import re
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

zodata = np.zeros([nsx,nt],dtype='float32')
zocrds = []
for isx in range(nsx):
  sidxs = asrcx == srcx[isx]
  recs  = arecx[sidxs]
  zoidx = recs == srcx[isx]
  isht  = data[sidxs,:]
  zodata[isx,:] = isht[zoidx,:]

ns = int(1/dt)
zodata[:,:ns] = 0.0

#plt.figure()
#plt.imshow(zodata.T,cmap='gray',interpolation='sinc',vmin=-0.01,vmax=0.01,aspect='auto')
#plt.show()

dzo = srcx[1] - srcx[0]; ozo = srcx[0]

sep = seppy.sep()
sep.write_file("sigsbee_zodata.H",zodata.T,os=[0.0,ozo],ds=[dt,dzo])

fdat.close(); fvel.close()

