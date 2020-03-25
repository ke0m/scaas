"""
Default geometry for synthetics
Sources and receivers on the surface
and distributed evenly across the surface
@author: Joseph Jennings
@version: 2020.03.23
"""
import numpy as np
import scaas.scaas2dpy as sca2d
import matplotlib.pyplot as plt

class defaultgeom:
  """ 
  Functions for modeling and imaging with a 
  standard synthetic source receiver geometry 
  """
  def __init__(self,nx,dx,nz,dz,nsx,osx,dsx,srcz=0.0,nrx=None,orx=0.0,drx=1.0,recz=0.0,alpha=0.99,bx=25,bz=25):
    """
    Creates a default geometry object for scalar acoustic 2D wave propagation

    Parameters:
      nx    - Number of x samples of the velocity model
      dx    - x sampling of the velocity model
      nz    - Number of z samples of the velocity model
      dz    - z sampling of the velocity model
      nsx   - Total number of sources
      osx   - x-sample coordinate of first source
      dsx   - Spacing between sources in samples
      srcz  - Constant depth of the sources [0.0]
      nrx   - Total number of receivers [One for every gridpoint]
      orx   - x-sample coordinate of first receiver [0.0]
      drx   - Spacing between receivers in samples [1.0]
      recz  - Constant depth of the receivers [0.0]
      alpha - Damping parameter of absorbing layer [0.99]
      bx    - Size of absorbing layer in the x direction [25]
      bz    - Size of absorbing layer in the z direction [25]

    Returns:
      a defaultgeom object
    """
    # Get padded size of model
    self.__nx = nx; self.__dx = dx
    self.__nz = nz; self.__dz = dz
    self.__bx = bx; self.__bz = bz
    self.__nxp = nx + 2*bx + 10
    self.__nzp = nz + 2*bz + 10

    # Make source and receiver coordinates
    self.allsrcx, self.allsrcz, self.nsrc = self.make_src_coords(nsx,osx,dsx,srcz)
    self.allrecx, self.allrecz, self.nrec = self.make_rec_coords(nrx,orx,drx,recz)

    # Boundary parameters
    self.__alpha = alpha

    # Image parameters (to be set by wem)
    self.rnh = None; self.oh = None; self.dh = None

  def model_fulldata(self,vel,wav,dtd,dtu=0.001,nthrds=4,verb=False):
    """
    Creates synthetic pressure data

    Parameters:
      vel    - Velocity model
      wav    - Wavelet (source time function). The sampling of the wavefield must be dtu
      dtd    - Output data sampling
      dtu    - Sampling of wavelet and wavefield
      nthrds - Number of OpenMP threads used for modeling
      verb   - Verbosity flag

    Returns:
      A numpy array of pressure data of shape [nsx,ntd,nrx]
    """
    if(vel.shape[1] != self.__nx or vel.shape[0] != self.__nz):
      raise Exception("Velocity must have same shape as passed to constructor")

    # Pad the velocity model
    velp = self.pad_model(vel)

    # Temporal axis
    ntu = wav.shape[0]
    fact = int(dtd/dtu); ntd = int(ntu/fact)

    # Build the wave propagation object
    sca = sca2d.scaas2d(ntd, self.__nxp, self.__nzp,        # Sizes
                        dtd, self.__dx, self.__dz,dtu,      # Samplings
                        self.__bx, self.__bz, self.__alpha) # Absorbing boundary
    # Create input wavelet array
    allsrcs = np.zeros([self.__nsx,1,ntu],dtype='float32')  # One source per shot (same length as wavefield)
    for isx in range(self.__nsx):
      allsrcs[isx,0,:] = wav[:]
    # Create output data
    allshot = np.zeros([self.__nsx,ntd,self.__nrx],dtype='float32')
    # Forward modeling
    sca.fwdprop_multishot(allsrcs,self.allsrcx,self.allsrcz,self.nsrc,    # Source information
                          self.allrecx,self.allrecz,self.nrec,            # Receiver information
                          self.__nsx,velp,allshot,nthrds,verb)          # Velocity and output

    return allshot

  def model_lindata(vel,dvel,wav,dtd,dtu=0.001,nthrds=4,verb=False):
    """
    Creates synthetic linearized (born) pressure data

    Parameters:
      vel    - Velocity model
      dvel   - Velocity perturbation
      wav    - Input wavelet (same length as wavefield)
      dtd    - Output data sampling
      dtu    - Wave propagation sampling (same as wavelet) [0.001]
      nthrds - Number of OpenMP threads [4]
      verb   - Verbosity flag [False]

    Returns
      A numpy array of linearized born data [nsx,ntd,nrx]
    """
    if(vel.shape[1] != self.__nx or vel.shape[0] != self.__nz):
      raise Exception("Velocity must have same shape as passed to constructor")

    if(vel.shape[1] != dvel.shape[1] or vel.shape[0] != dvel.shape[0]):
      raise Exception("Velocity and perturbation must have same shape")

    # Pad the models
    velp = self.pad_model(vel); dvelp = self.pad_model(dvel)

    # Temporal axis
    ntu = wav.shape[0]
    fact = int(dtd/dtu); ntd = int(ntu/fact)

    # Build the wave propagation object
    sca = sca2d.scaas2d(ntd, self.__nxp, self.__nzp,        # Sizes
                        dtd, self.__dx, self.__dz,dtu,      # Samplings
                        self.__bx, self.__bz, self.__alpha) # Absorbing boundary
    # Create input wavelet array
    allsrcs = np.zeros([self.__nsx,1,ntu],dtype='float32')  # One source per shot (same length as wavefield)
    for isx in range(nsx):
      allsrcs[isx,0,:] = src[:]
    # Create output data
    allshot = np.zeros([self.__nsx,ntd,self.__nrx],dtype='float32')

    sca.brnfwd(allsrcs,self.allsrcx,self.allsrcz,self.nsrc,   # Source information
               self.allrecx,self.allrecz,self.nrec,           # Receiver information
               self.__nsx,velp,dvelp,allshot,nthrds,verb)   # Velocity and output

    return allshot

  def wem(vel,wav,dat,dtd,dtu=0.001,nh=None,nthrds=4,verb=False):
    """
    Wave equation migration (WEM) or Image reconstruction
    
    Parameters
      vel    - migration velocity
      wav    - Input wavelet (source time function) same size as wavefield
      dtd    - Output data sampling
      dtu    - Wave propagation time step (sampling). Same as wavelet sampling [0.001]
      nh     - Number of subsurface offsets (output will be 2*nh + 1) [None]
      nthrds - Number of OpenMP threads to use [4]
      verb   - Verbosity flag [False]

    Returns
      A numpy array containing an image (extended if nh provided) of dimension [nh,nz,nx]
    """
    if(vel.shape[1] != self.__nx or vel.shape[0] != self.__nz):
      raise Exception("Velocity must have same shape as passed to constructor")

    if(dat.shape[0] != self.__nsx or dat.shape[2] != self.__nrx):
      raise Exception("Data must have same shape as acqusition passed to constructor")

    # Pad the model and create the output image
    velp = self.pad_model(vel)
    if(nh == None):
      imgp = np.zeros([self.__nxp,self.__nzp],dtype='float32')
    else:
      self.rnh = 2*nh + 1; self.oh = -self.__dx*nh; self.dh = self.__dx
      imgp = np.zeros([rnh,self.__nxp,self.__nzp],dtype='float32')

    # Temporal axis
    ntu = wav.shape[0]
    fact = int(dtd/dtu); ntd = int(ntu/fact)

    # Build the wave propagation object
    sca = sca2d.scaas2d(ntd, self.__nxp, self.__nzp,        # Sizes
                        dtd, self.__dx, self.__dz,dtu,      # Samplings
                        self.__bx, self.__bz, self.__alpha) # Absorbing boundary
    # Create input wavelet array
    allsrcs = np.zeros([self.__nsx,1,ntu],dtype='float32')  # One source per shot (same length as wavefield)
    for isx in range(nsx):
      allsrcs[isx,0,:] = src[:]

    # Form the image
    if(nh == None):
      sca.brnadj(allsrcs,self.allsrcx,self.allsrcz,self.nsrc, # Source information
                 self.allrecx,self.allrecz,self.nrec,         # Receiver information
                 nsx,velp,imgp,dat,nthrds,verb)               # Velocity and output image
      img = self.trunc_model(imgp)
    # Extended image
    else:
      sca.brnoffadj(allsrcs,self.allsrcx,self.allsrcz,self.nsrc, # Source information
                    self.allrecx,self.allrecz,self.nrec,         # Receiver information
                    nsx,velp,self.rnh,imgp,allshot,nthrds,verb)  # Velocity and output image
      img = self.trunc_model(imgp)

    return img

  def get_ext_axis(self):
    """ Returns the subsurface offset extension axis """
    return self.rnh, self.oh, self.dh

  def make_src_coords(self,nsx,osx,dsx,srcz=0.0):
    """ 
    Makes source coordinates assuming the default geometry 

    Parameters:
      nsx - Number of sources
      osx - x-sample coordinate of first source
      dsx - Spacing between sources in samples
    """
    self.__nsx = nsx
    osxp = osx + self.__bx + 5; srczp = srcz + self.__bz + 5
    nsrc = np.ones(nsx,dtype='int32')               # One source 
    allsrcx = np.zeros([nsx,1],dtype='int32')  # for each experiment
    allsrcz = np.zeros([nsx,1],dtype='int32')  # (no blending)
    # All source x positions in one array
    srcs = np.linspace(osxp,osxp + (nsx-1)*dsx,nsx)
    for isx in range(nsx):
      allsrcx[isx,0] = int(srcs[isx])
      allsrcz[isx,0] = int(srczp)

    return allsrcx, allsrcz, nsrc

  def make_rec_coords(self,nrx,nsx,orx=0.0,drx=1.0,recz=0.0):
    """ 
    Makes the receiver coordinates assuming the default geometry 

    Parameters:
      nrx - Number of receivers
      orx - x-sample coordinate of first receiver
      drx - Spacing between receivers in samples
    """
    # Create receiver coordinates
    if(nrx == None): 
      nrx = self.__nx; orx = 0.0; drx = 1.0;      # Force orx=0.0 and drx=1.0 for this case
    self.__nrx = nrx
    orxp = orx + self.__bx + 5; reczp  = recz + self.__bz + 5 
    nrec = np.zeros(nrx,dtype='int32') + nrx       # All receivers
    allrecx = np.zeros([self.__nsx,nrx],dtype='int32')    # for each experiment
    allrecz = np.zeros([self.__nsx,nrx],dtype='int32')    # (stationary and always active)
    # Create all receiver positions
    recs = np.linspace(orxp,orxp + (nrx-1)*drx,nrx)
    for isx in range(self.__nsx):
      allrecx[isx,:] = (recs[:]).astype('int32')
      allrecz[isx,:] = np.zeros(len(recs),dtype='int32') + reczp

    return allrecx, allrecz, nrec

  def pad_model(self,vel):
    """ Pad the velocity/reflectivity model """
    # Pad for absorbing layer
    velp = np.pad(vel,((self.__bx,self.__bx),(self.__bz,self.__bz)),'edge')
    # Pad for laplacian stencil
    velp = np.pad(velp,((5,5),(5,5)),'constant')

    return velp.astype('float32')

  def trunc_model(vel):
    """ Truncate the velocity model """
    if(len(vel.shape) == 2):
      return vel[self.__bz+5:self.__nz+self.__bz+5,self.__bx+5:self.__nx+self.__bx+5]
    elif(len(vel.shape) == 3):
      return vel[:,self.__bz+5:self.__nz+self.__bz+5,self.__bx+5:self.__nx+self.__bx+5]

  def plot_acq(self,mod=None,**kwargs):
    """ Plots the acquisition on a velocity model """
    # Plot velocity model
    if(mod is None):
      mod  = np.zeros(self.__nx, self.__nz) + 2500.0
      modp = self.pad_model(mod)
    else:
      modp = self.pad_model(mod)
    vmin = np.min(mod); vmax = np.max(mod)
    fig = plt.figure(figsize=(kwargs.get('wbox',10),kwargs.get('hbox',10)))
    ax = fig.gca()
    im = ax.imshow(modp,extent=[0,self.__nxp,self.__nzp,0],vmin=vmin,vmax=vmax,cmap='jet')
    ax.set_xlabel(kwargs.get('xlabel','X (gridpoints)'),fontsize=kwargs.get('labelsize',14))
    ax.set_ylabel(kwargs.get('ylabel','Z (gridpoints)'),fontsize=kwargs.get('labelsize',14))
    ax.tick_params(labelsize=kwargs.get('labelsize',14))
    # Get all source positions
    plt.scatter(self.allrecx[0,:],self.allrecz[0,:],c='tab:green',marker='v')
    plt.scatter(self.allsrcx[:,0],self.allsrcz[:,0],c='tab:red',marker='*')
    plt.show()

