# Build point scatterer models for imaging
import numpy as np

def find_optimal_sizes(n,j,nb):
  """ Finds the optimal size for a provided j along a 
  particular axis
  """
  b = []; e = []; 
  b.append(0)
  space = n 
  for k in range(nb-1):
    sample = np.ceil(float(space)/float(nb-k-1))
    e.append(sample + b[k] - 1)
    if(k != nb-2): b.append(e[k] + 1)
    space -= sample
  return b,e 

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

