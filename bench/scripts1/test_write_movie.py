import inpout.seppy as seppy

sep = seppy.sep([])

# Read in model movie
maxes,mmov = sep.read_file(None,ifname="marm1550mmov.H")
mmov = mmov.reshape(maxes.n,order='F')

# Read in objective function movie
faxes,ofn = sep.read_file(None,ifname="marm1550ofn.H")

# Write first iteration
nz = maxes.n[0]; nx = maxes.n[1]; niter = maxes.n[2]
naxes = seppy.axes([nz,nx],[0.0,0.0],[1.0,1.0])
sep.write_file(None,naxes,mmov[:,:,0],ofname='mymmov.H')

nfaxes = seppy.axes([1],[0.0],[1.0])
nfaxes.ndims = 0
sep.write_file(None,nfaxes,ofn[0],ofname="myofn.H")

for iiter in range(1,niter):
  # Write subsequent iterations
  sep.append_to_movie(None,naxes,mmov[:,:,iiter],iiter+1,ofname='mymmov.H')
  sep.append_to_movie(None,nfaxes,ofn[iiter],iiter+1,ofname='myofn.H')
