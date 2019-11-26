import scaas.scaas2dpy as sca2d
import numpy as np
from scaas.gradtaper import build_taper

class fwi:
  """ Functions for computing gradient and value of FWI objective functions """
  def __init__(self,maxes,saxes,allsrcs,daxes,dat,acqdict,prpdict,tpdict=None,nthrd=1):
    """ Constructor """
    ## Dimensions
    # Get model dimensions
    self.nx = maxes.n[1]; self.dx = maxes.d[1]
    self.nz = maxes.n[0]; self.dz = maxes.d[0]
    # Source dimensions
    self.ntu = saxes.n[0]; self.dtu = saxes.d[0]
    # Get data dimensions
    self.nt  = daxes.n[0]; self.dt  = daxes.d[0]
    self.nrx = daxes.n[1]; self.drx = daxes.d[1]
    self.nsx = daxes.n[2]; self.dsx = daxes.d[2]
    ## Acquisition
    # Source positions
    self.nsrc    = acqdict['nsrc']
    self.allsrcx = acqdict['allsrcx']
    self.allsrcz = acqdict['allsrcz']
    # Receiver positions
    self.nrec    = acqdict['nrec']
    self.allrecx = acqdict['allrecx']
    self.allrecz = acqdict['allrecz']
    # Number of examples (usually sources)
    self.nex     = acqdict['nex']
    ## Propagation parameters
    # Boundaries
    self.bx = prpdict['bx']; self.bz = prpdict['bz']
    self.alpha = prpdict['alpha']
    # Number of threads
    self.nthrd = nthrd
    # Create wave propagation object
    self.sca = sca2d.scaas2d(self.nt, self.nx, self.nz, self.dt, self.dx, self.dz, self.dtu,
                             self.bx, self.bz, self.alpha)
    ## Input arrays
    # Get source
    self.allsrcs = allsrcs
    # Get data
    self.odat = dat
    self.mdat = np.zeros(dat.shape,dtype='float32')
    ## Gradient taper
    # Build taper
    if(tpdict == None):
      _,self.gtap2 = build_taper(self.nx,self.nz,0,0)
    else:
      _,self.gtap2 = build_taper(self.nx,self.nz,tpdict['izt'],tpdict['izb'])


  def gradientL2(self,velcur,grad):
    """ Gradient of L2 FWI Objective function """
    ## Forward modeling for all shots for current model
    self.mdat[:] = np.zeros(self.odat.shape,dtype='float32')
    self.sca.fwdprop_multishot(self.allsrcs, self.allsrcx, self.allsrcz, self.nsrc,
                          self.allrecx, self.allrecz, self.nrec, 
                          self.nex, velcur, self.mdat, self.nthrd)
    
    ## Compute adjoint source
    res = self.mdat - self.odat
    asrc = -res
    
    ## Gradient for all shots
    gradutap = np.zeros([self.nz,self.nx],dtype='float32'); grad[:] = 0.0
    self.sca.gradient_multishot(self.allsrcs, self.allsrcx, self.allsrcz, self.nsrc,
                                asrc        , self.allrecx, self.allrecz, self.nrec,
                                self.nex, velcur, gradutap, self.nthrd)
    # Apply taper near the source positions
    grad[:] = self.gtap2*gradutap

    return 0.5*np.dot(res.flatten(),res.flatten())

  def get_moddat(self):
    """ Returns the modeled data for the current model velcur.
        Must be called after gradientL2 otherwise data are overwritten.
    """
    return np.transpose(self.mdat,(1,2,0))

