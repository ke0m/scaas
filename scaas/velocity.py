"""
Velocity utility functions for imaging
@author: Joseph Jennings
@version: 2020.08.10
"""
import numpy as np
import scaas.noise_generator as noise_generator
import scipy.ndimage as flt
from genutils.rand import randfloat
from scaas.trismooth import smooth

def find_optimal_sizes(n,j,nb):
  """
  Finds the optimal size for a provided j along a
  particular axis
  """
  b = []; e = []
  b.append(0)
  space = n
  for k in range(nb-1):
    sample = np.ceil(float(space)/float(nb-k-1))
    e.append(sample + b[k] - 1)
    if(k != nb-2): b.append(e[k] + 1)
    space -= sample
  return b,e

def distance(pt1,pt2):
  """ Compute the distance between two points """
  return np.linalg.norm(np.asarray(pt1)-np.asarray(pt2))

def create_ptscatmodel(nz,nx,j1,j2,verb=False):
  """ Creates a point scatterer model """
  # Calculate the number of blocks in each dimension
  # Make sure the size of the block is odd
  nb = np.zeros(2,dtype=int)
  nb[0] = nz/j1
  nb[1] = nx/j2
  # Block halfway points
  hb = nb/2
  # Get the beginning and ending of each block
  p1 = np.arange(0,nz,j1)
  p2 = np.arange(0,nx,j2)

  # Calculate block coordinates for z
  e1 = []; b1 = []
  for k in range(len(p1)-1):
    b1.append(p1[k]); e1.append(p1[k+1]-1)
  b1.append(p1[-1]); e1.append(nz-1)
  if(verb): print("Z blocks:",(b1,e1))

  # Calculate block coordinates for x
  e2 = []; b2 = []
  for k in range(len(p2)-1):
    b2.append(p2[k]); e2.append(p2[k+1]-1)
  b2.append(p2[-1]); e2.append(nz-1)
  if(verb): print("X blocks:",(b2,e2))

  # Create the output model
  scats = np.zeros([nz,nx],dtype='float32')
  for ib1 in range(nb[0]):
    for ib2 in range(nb[1]):
      ct1 = int(b1[ib1] + j1/2 + 1)
      ct2 = int(b2[ib2] + j2/2 + 1)
      scats[ct1,ct2] = 1.0
      if(verb): print("Block %d %d: (%d,%d)"%(ib1,ib2,ct1,ct2))

  return scats

def create_randptscatmodel(nz,nx,npts,mindist):
  """ Creates a model with randomly distributed point scatterers """
  scats = np.zeros([nz,nx],dtype='float32')
  pts = []; k = 0
  while(len(pts) < npts):
    # Create a coordinates
    pt = []
    pt.append(np.random.randint(10,nz-10))
    pt.append(np.random.randint(10,nx-10))
    if(k == 0):
      scats[pt[0],pt[1]] = 1.0
      pts.append(pt)
    else:
      keeppoint = True
      for opt in pts:
        if(distance(pt,opt) < mindist):
          keeppoint = False
          break
      if(keeppoint == True):
        scats[pt[0],pt[1]] = 1.0
        pts.append(pt)
    k += 1

  return scats

def create_randomptb(nz,nx,romin,romax,nptsz=1,nptsx=1,octaves=4,period=80,Ngrad=80,persist=0.2,ncpu=2):
  """
  Creates a low wavenumber perturbation given minimum rho
  and maximum rho values
  """
  noise = noise_generator.perlin(x=np.linspace(0,nptsx,nx), y=np.linspace(0,nptsz,nz), octaves=octaves,
      period=period, Ngrad=Ngrad, persist=persist, ncpu=ncpu)
  noise -= np.min(noise)
  n = ((romax - romin) + np.max(noise*romin))/(np.max(noise))
  return (noise*(n-romin) + romin).astype('float32')

def create_randomptb_loc(nz,nx,romin,romax,naz,nax,cz,cx,
                         nptsz=1,nptsx=1,octaves=4,period=80,Ngrad=80,persist=0.2,ncpu=2,sigma=20):
  """
  Creates a low wavenumber perturbation given minimum rho
  and maximum rho values
  """
  noise = noise_generator.perlin(x=np.linspace(0,nptsx,nax), y=np.linspace(0,nptsz,naz), octaves=octaves,
      period=period, Ngrad=Ngrad, persist=persist, ncpu=ncpu)
  noise -= np.min(noise)
  n = ((romax - romin) + np.max(noise*romin))/(np.max(noise))
  noiseout = noise*(n-romin) + romin

  #nz = naz + p1 + p2
  # cz = p1 + naz
  #pz = int((nz-naz)/2); px = int((nx-nax)/2)
  pz1 = cz - int(naz/2)
  if(naz%2 != 0): pz1 -= 1
  if(pz1 < 0):
    cz = int(naz/2)
  pz2 = nz - cz - int(naz/2)
  #print(pz1,int(naz/2),pz2)
  px1 = cx - int(nax/2)
  if(nax%2 != 0): px1 -= 1
  if(px1 < 0):
    cx = int(nax/2)
  px2 = nx - cx - int(nax/2)
  noisep   = np.pad(noiseout,((pz1,pz2),(px1,px2)),'constant',constant_values=1)
  noisepsm = flt.gaussian_filter(noisep,sigma=sigma)

  return noisepsm.astype('float32')

def create_constptb_loc(nz,nx,ptb,naz,nax,cz,cx,rectx=20,rectz=20):
  """
  Creates a constant perturbation of size nax by nax and at position
  (cz,cx)

  Parameters
    nz    - number of depth samples of output velocity model
    nx    - number of lateral samples of output velocity model
    ptb   - percent perturbation
    naz   - number of depth samples of perturbation 
    nax   - number of lateral samples of perturbation
    cz    - z center position of anomaly
    cx    - x center position of anomaly
    rectx - number of points to smooth in x [20]
    rectz - number of points to smooth in z [20]

  Returns a model [nz,nx] with the anomaly positioned at (cz,cx) in the model
  """
  velc = np.zeros([naz,nax],dtype='float32') + ptb
  pz1 = cz - int(naz/2)
  if(naz%2 != 0): pz1 -= 1
  if(pz1 < 0): 
    cz = int(naz/2)
  pz2 = nz - cz - int(naz/2)
  px1 = cx - int(nax/2)
  if(nax%2 != 0): px1 -= 1
  if(px1 < 0): 
    cx = int(nax/2)
  px2 = nx - cx - int(nax/2)
  velcp = np.pad(velc,((pz1,pz2),(px1,px2)),'constant',constant_values=1)
  return smooth(velcp,rect1=rectx,rect2=rectz)

def create_randomptbs_loc(nz,nx,nptbs,romin=0.95,romax=1.06,minnaz=150,maxnaz=300,minnax=150,maxnax=300,mindist=200,
    mincz=None,mincx=None,maxcz=None,maxcx=None,nptsz=1,nptsx=1,octaves=4,period=80,Ngrad=80,
    persist=0.2,ncpu=1,sigma=20):
  """
  Creates randomly positioned velocity anomalies in a velocity model

  Parameters
    nz      - number of z samples of velocity model
    nx      - number of x samples of velocity model
    romins  - a list of minimum % magnitudes of velocity perturbation
    romaxs  - a list of maximum % magnitudes of velocity perturbation
    nazs    - list of average sizes of anomalies in z-direction
    naxs    - list of average sizes of anomalies in x-direction
    mindist - minimum distance between centers of anomalies
    mincx   - minimum x sample to place center of anomaly [0]
    mincz   - minimum z sample to place center of anomaly [0]
    maxcx   - maximum x sample to place center of anomaly [0]
    maxcz   - maximum z sample to place center of anomaly [0]
  """
  # Get ranges
  if(mincz is None): mincz = 50
  if(maxcz is None): maxcz = nz/2
  if(mincx is None): mincx = 20
  if(maxcx is None): maxcx = nz-20
  # Find centers
  ctrs = []; k = 0
  while(len(ctrs) < nptbs):
    ctr = []
    ctr.append(np.random.randint(mincz,maxcz))
    ctr.append(np.random.randint(mincx,maxcx))
    if(k == 0):
      ctr.append(ctr)
    else:
      keeppoint = True
      for ict in ctrs:
        if(distance(ctr,ict) < mindist):
          keeppoint = False
          break
      if(keeppoint == True):
        ctrs.append(ctr)
    k += 1

  # Create each perturbation
  velout = np.ones([nz,nx],dtype='float32')
  for iptb in range(nptbs):
    # Create naz and nax from the min and the max
    nax = np.random.randint(minnax,maxnax)
    naz = np.random.randint(minnaz,maxnaz)
    # Create romin and romax from the min and max
    if(romin < 1.0 and romax > 1.0 and nptbs > 1):
      if(np.random.choice([0,1])):
        iromin = romin + randfloat(0.001,0.01); iromax = 1.0 + randfloat(0.001,0.01)
      else:
        iromax = romax + randfloat(0.001,0.01); iromin = 1.0 - randfloat(0.001,0.01)
    else:
      iromin = romin; iromax = romax
    ptb = create_randomptb_loc(nz,nx,iromin,iromax,naz,nax,ctrs[iptb][0],ctrs[iptb][1],
                               nptsz=nptsz,nptsx=nptsx,octaves=octaves,period=period,Ngrad=Ngrad,
                               persist=persist,ncpu=1,sigma=sigma)
    velout *= ptb

  return velout

def create_layered(nz,nx,dz,dx,z0s=[],vels=[],flat=True,
    npts=2,octaves=3,period=80,Ngrad=80,persist=0.6,scale=200,ncpu=1,interp='lint'):
  """
  Creates a layered velocity and reflectivity model.
  Can create random undulation in the layers.
  """
  # Check the input arguments
  assert(any(z0s) < nz or any(z0s) > 0), "z0 out of range"
  nref = len(z0s); nvels = len(vels)
  assert(nvels == nref+1), "nvels != nz0s+1"
  # Convert the lists to numpy arrays
  z0sar  = np.asarray(z0s)
  velsar = np.asarray(vels)
  # Outputs
  ovel = np.zeros([nz,nx],dtype='float32')
  lyrs = np.zeros([nz,nx],dtype='float32')
  refs = []

  # First put in the reflector positions
  for iref in range(nref):
    # Create flat layer
    spk = np.zeros(nz)
    spk[z0s[iref]] = 1
    rpt = np.tile(spk,(nx,1)).T
    # Calculate shifts
    if(flat == False):
      shp = noise_generator.perlin(x=np.linspace(0,npts,nx), octaves=octaves, period=80, Ngrad=80, persist=persist, ncpu=2)
      shp -= np.mean(shp); shp *= scale
      ishp = shp.astype(int)
    else:
      ishp = np.zeros(nx,dtype=np.int)
    pos = ishp + z0s[iref]
    refs.append(pos)
    # Put in layer
    if(interp == 'lint'):
      lyrs += np.array([flt.interpolation.shift(rpt[:,ix],shp[ix],order=1) for ix in range(nx)]).T
    else:
      lyrs += np.array([np.roll(rpt[:,ix],ishp[ix]) for ix in range(nx)]).T

  # Fill in the velocity
  for iref in range(nref):
    if(iref == 0):
      ref = refs[iref]
      for ix in range(nx):
        ovel[0:ref[ix],ix] = vels[iref]
    else:
      ref1 = refs[iref-1]; ref2 = refs[iref]
      for ix in range(nx):
        ovel[ref1[ix]:ref2[ix],ix] = vels[iref]

  # Get the last layer
  ref = refs[-1]
  for ix in range(nx):
    ovel[ref[ix]:,ix] = vels[-1]

  return ovel,lyrs

def salt_mask(img,vel,saltvel,thresh=0.95,rectx=30,rectz=30):
  """
  Masks the image based on the salt velocity

  Parameters
    img     - input image (can be extended or not) [nhy,nhx,nz,ny,nx]
    vel     - input migration velocity model [nz,ny,nx]
    saltvel - salt velocity for masking
    thresh  - mask threshold [0.95]
    rectx   - amount of smoothing to be applied to mask along x
    rectz   - amount of smoothing to be applied to mask along z

  Returns mask and masked image (msk,maskedimg)
  """
  # Create the mask
  idx = vel >= saltvel
  msk = np.ascontiguousarray(np.copy(vel)).astype('float32')
  msk[idx] = 0.0 
  msk[~idx] = 1.0 
  if(len(msk.shape) == 3):
    mskw = msk[:,0,:]
  else:
    mskw = msk

  # Smooth and threshold the mask
  smmsk = smooth(mskw,rect1=30,rect2=30)
  idx2 = smmsk > 0.95
  smmsk[idx2] = 1.0 
  smmsk[~idx2] = 0.0 
  smmsk2 = smooth(smmsk,rect1=2,rect2=2)

  # Apply the mask to the image
  imgo = np.zeros(img.shape,dtype='float32')
  if(len(img.shape) == 5):
    nhx = img.shape[1]
    for ihx in range(nhx):
      imgo[0,ihx,:,0,:] = smmsk2*img[0,ihx,:,0,:]
  elif(len(img.shape) == 3):
    imgo = smmsk2*img[:,0,:]
  elif(len(img.shape) == 2):
    imgo = smmsk2*img

  return smmsk2,imgo

