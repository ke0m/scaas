import scaas.scaas2dpy as sca2d
import numpy as np
import matplotlib.pyplot as plt

class fwi:
  """ Functions for computing gradient and value of FWI objective functions """
  def __init__(self,maxes,saxes,allsrcs,daxes,dat,acqdict,prpdict,nthreads):
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
    self.nrec    = acqdict['nsrc']
    self.allrecx = acqdict['allrecx']
    self.allrecz = acqdict['allrecz']
    # Number of examples (usually sources)
    self.nex     = acqdict['nex']
    ## Propagation parameters
    # Boundaries
    self.bx = prpdict['bx']; self.bz = prpdict['bz']; 
    self.alpha = prpdict['alpha']
    # Number of threads
    self.nthreads = nthreads
    # Create wave propagation object
    self.sca = sca2d.scaas2d(self.nt, self.nx, self.nz, self.dt, self.dx, self.dz, self.dtu,
                             self.bx, self.bz, self.alpha)
    # Get source
    self.allsrcs = allsrcs
    # Get data
    self.dat = dat

  def gradientL2(self,velcur,grad):
    """ Gradient of L2 FWI Objective function """
    # Forward modeling for all shots for current model
    moddat = np.zeros([self.nsx,self.nt,self.nrx],dtype='float32')
    self.sca.fwdprop_multishot(self.allsrcs, self.allsrcx, self.allsrcz, self.nsrc,
                          self.allrecx, self.allrecz, self.nrec, 
                          self.nex, velcur, moddat, self.nthreads)
    
    ## Compute adjoint source
    res = moddat - self.dat
    asrc = -res
    
    # Gradient for all shots
    self.sca.gradient_multishot(self.allsrcs, self.allsrcx, self.allsrcz, self.nsrc,
                                asrc        , self.allrecx, self.allrecz, self.nrec,
                                self.nex, velcur, grad, self.nthreads)

    return 0.5*np.dot(res.flatten(),res.flatten())

