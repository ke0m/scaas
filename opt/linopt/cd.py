"""
Functions for a least-squares conjugate directions solver

@author: Joseph Jennings
@version: 2020.06.17
"""
import numpy as np
from opt.linopt.opr8tr import operator
from opt.linopt.combops import colop
from opt.optqc import optqc
from genutils.ptyprint import create_inttag

def cd(op,dat,mod0,regop=None,rdat=None,grdop=None,shpop=None,eps=None,niter=None,toler=None,
       optqc=None,objs=None,mods=None,grds=None,ress=None,verb=True):
  """
  Conjugate direction solver as described in GIEE by Jon Claerbout 
  Minimizes the objective 
  
       J(m) = ||Lm-d||^2_2 + eps||Am-q||^2_2

  where L is a linear operator, m is a model vector, d is a data vector,
  A is a regularization operator, q is a regularization vector
  and eps is a regularization parameter.

  Parameters
    op    - the operator (L)
    dat   - the input data vector (d)
    mod0  - the initial model vector (m)
    regop - the regularization operator (A) [None]
    rdat  - the regularization data (q) [None]
    eps   - a regularization parameters [None]
    grdop - a gradient operator (e.g., a gradient taper or smoother) [None]
    shpop - a shaping regularization operator (e.g., triangular smoother)
    niter - number of iterations for which to run the optimization [None]
    toler - tolerance to reach before terminating the optimization
    optqc - a optimization qc object for writing to SEP files or PNG/PDF figures [None]
    objs  - an empty list for saving the objective function values [None]
    mods  - an empty list for saving the model iterates [None]
    grds  - an empty list for saving the gradients [None]
    ress  - an empty list for saving the residuals [None]
    verb  - whether to print solver output at each iteration [True]

  Returns the estimated model (same shape as mod0)
  """
  # Form the column op if the regularization operator is set
  if(regop is not None):
    if(eps is None):
      raise Exception("Please provide an epsilon value to run regularized solver")
    # Create operators and data
    ops  = [op,regop]
    dats = [dat,eps*rdat]
    # Make dimensions
    ddim = {}; rdim = {}
    ddim['ncols'] = mod0.shape; ddim['nrows'] = dat.shape
    rdim['ncols'] = mod0.shape; rdim['nrows'] = rdat.shape
    dims = [ddim,rdim]
    # Total operator
    epss = np.asarray([1.0,eps])
    top = colop(ops,dims,epss)
  else:
    top = op; dats = dat
  # Make copy of the model
  mod0i = np.copy(mod0)
  # Run the solver
  if(shpop is not None):
    if(eps   is None): eps   = 1.0
    if(toler is None): toler = 1.e-6 
    run_cgshape(top,dats,mod0i,shpop,eps,niter,toler,objs,mods,grds,ress,optqc,verb)
  elif(niter is not None):
    run_niter(top,dats,mod0i,niter,grdop,objs,mods,grds,ress,optqc,verb)
  elif(toler is not None):
    run_toler(top,dats,mod0i,toler,grdop,objs,mods,grds,ress,optqc,verb)
  else:
    raise Exception("Must specify niter or tolerance to run solver")

  return mod0i

def run_niter(op,dat,mod,niter,grdop,objs,mods,grds,ress,optqc,verb):
  """ Runs conjugate direction solver for niter iterations """
  # Temporary data space arrays
  if(isinstance(dat,list)):
    res = []; drs = []; dsz = []
    for idat in dat:
      res.append(np.zeros(idat.shape,dtype='float32'))
      drs.append(np.zeros(idat.shape,dtype='float32'))
      dsz.append(idat.shape)
  else:
    res = np.zeros(dat.shape,dtype='float32')
    drs = np.zeros(dat.shape,dtype='float32')
    dsz = dat.shape

  # Temporary model space arrays
  if(isinstance(mod,list)):
    grd = []; tap = []; msz = []
    for imod in mod:
      grd.append(np.zeros(imod.shape,dtype='float32'))
      tap.append(np.zeros(imod.shape,dtype='float32'))
      msz.append(imod.shape)
  else:
    grd = np.zeros(mod.shape,dtype='float32')
    tap = np.zeros(mod.shape,dtype='float32')
    msz = mod.shape

  # Create a stepper object
  stpr = cdstep(msz,dsz)

  # Loop over all iterations
  for iiter in range(niter):
    # First compute the objective function
    op.forward(False,mod,res)
    scale_add(res,1.0,dat,-1.0)
    f0 = (0.5)*gdot(res,res)
    # Compute the gradient
    op.adjoint(False,grd,res)
    # Process the gradient if desired
    if(grdop is not None):
      grdop.forward(False,grd,tap)
    else:
      tap = grd
    # Compute the data space gradient
    op.forward(False,tap,drs)
    # Compute the step length and update
    if(not stpr.step(mod,tap,res,drs)): break
    # Reevaluate objective function and output
    f1 = (0.5)*gdot(res,res)
    if(f1 >= f0):
      print("Objective function did not reduce, terminating solver")
      break
    # Save results to provided lists
    save_results(mod,mods,tap,grds,res,ress,f1,objs)
    # Save to SEPlib file or image if desired
    if(optqc is not None):
      optqc.output(f1,mod,tap,res)
    if(verb):
      print("iter=%s objf=%.6f gnrm=%.6f"%(create_inttag(iiter+1,niter),f1,np.linalg.norm(tap)))

def run_toler(op,dat,mod,toler,grdop,mods,objs,grds,ress,optqc,verb):
  """ Runs conjugate direction solver until a tolerance is reached """
  # Temporary data space arrays
  if(isinstance(dat,list)):
    res = []; drs = []; dsz = []
    for idat in dat:
      res.append(np.zeros(idat.shape,dtype='float32'))
      drs.append(np.zeros(idat.shape,dtype='float32'))
      dsz.append(idat.shape)
  else:
    res = np.zeros(dat.shape,dtype='float32')
    drs = np.zeros(dat.shape,dtype='float32')
    dsz = dat.shape

  # Temporary model space arrays
  if(isinstance(mod,list)):
    grd = []; tap = []; msz = []
    for imod in mod:
      grd.append(np.zeros(imod.shape,dtype='float32'))
      tap.append(np.zeros(imod.shape,dtype='float32'))
      msz.append(imod.shape)
  else:
    grd = np.zeros(mod.shape,dtype='float32')
    tap = np.zeros(mod.shape,dtype='float32')
    msz = mod.shape

  # Create a stepper object
  stpr = cdstep(msz,dsz)

  # Loop until tolerance is reached
  f1 = 1.0 + toler; iiter = 0
  while(f1 > rtol):
    # First compute the objective function
    op.forward(False,mod,res)
    scale_add(res,1.0,dat,-1.0)
    f0 = (0.5)*gdot(res,res)
    # Compute the gradient
    op.adjoint(False,grd,res)
    # Process the gradient if desired
    if(grdop is not None):
      grdop.forward(False,grd,tap)
    else:
      tap = grd
    # Compute the data space gradient
    op.forward(False,tap,drs)
    # Compute the step length and update
    if(not stpr.step(mod,tap,res,drs)): break
    # Reevaluate objective function and output
    f1 = (0.5)*gdot(res,res)
    if(f1 >= f0):
      print("Objective function did not reduce, terminating solver")
      break
    # Save results to provided lists
    save_results(mod,mods,tap,grds,res,ress,f1,objs)
    # Save to SEPlib file or image if desired
    if(optqc is not None):
      optqc.output(f1,mod,tap,res)
    if(verb):
      print("iter=%s objf=%.6f gnrm=%.6f"%(create_inttag(iiter+1,10000),f1,np.linalg.norm(tap)))
    iiter += 1

def run_cgshape(op,dat,mod,shpop,eps,niter,toler,objs,mods,grds,ress,optqc,verb):
  """ Sergey Fomel's conjugate gradient with shaping regularization """
  ## Allocate memory
  # Model and residual
  pod = np.zeros(mod.shape,dtype='float32')
  res = np.zeros(dat.shape,dtype='float32')
  # Gradients
  grp = np.zeros(mod.shape,dtype='float32')
  grm = np.zeros(mod.shape,dtype='float32')
  grr = np.zeros(dat.shape,dtype='float32')
  # Search directions
  srp = np.zeros(mod.shape,dtype='float32')
  srm = np.zeros(mod.shape,dtype='float32')
  srr = np.zeros(dat.shape,dtype='float32')

  # Epsilon scaling
  eps2 = eps*eps

  # Compute the initial residual
  res[:] = -dat[:]
  op.forward(True,mod,res)

  dg = g0 = gnp = 0.0
  r0 = gdot(res,res)
  if(r0 == 0.0):
    print("Residual is zero: r0=%f"%(r0))
    return

  # Iteration loop
  for iiter in range(niter):
    grp[:] = eps*pod[:]; grm = -eps*mod[:]

    # Compute the traditional gradient
    op.adjoint(True,grm,res)

    # Symmetrized shaping operator
    shpop.adjoint(True,grp,grm)
    shpop.forward(False,grp,grm)

    # Data space gradient
    op.forward(False,grm,grr)

    gn = gdot(grp,grp)

    if(iiter == 0):
      # Use only current gradient for first iteration
      g0 = gn; srp[:] = grp[:]; srm[:] = grm[:]; srr[:] = grr[:]
    else:
      alpha = gn/gnp
      dg    = gn/g0

      if(alpha < toler or dg < toler):
        if(verb): print("converged in %d iterations, alpha=%f gd=%f"%(iiter,alpha,dg))
        break

      scale_add(grp,1.0,srp,alpha); swap(srp,grp)
      scale_add(grm,1.0,srm,alpha); swap(srm,grm)
      scale_add(grr,1.0,srr,alpha); swap(srr,grr)

    beta = gdot(srr,srr) + eps*(gdot(srp,srp) - gdot(srm,srm))

    # Compute step length
    alpha = -gn/beta

    # Update model and residual
    scale_add(pod,1.0,srp,alpha)
    scale_add(mod,1.0,srm,alpha)
    scale_add(res,1.0,srr,alpha)

    # Verbosity
    rout = gdot(res,res)/r0
    # Save results to provided lists
    save_results(mod,mods,grm,grds,res,ress,rout,objs)
    # Save to SEPlib file or image if desired
    if(optqc is not None):
      optqc.output(rout,mod,grm,res)
    if(verb):
      print("iter=%s res=%.6f grd=%.6f"%(create_inttag(iiter+1,niter),rout,dg))

    gnp = gn

def save_results(mod,mods,grd,grds,res,ress,obj,objs):
  """ Saves the iterates to provided lists """
  if(mods is not None):
    mods.append(np.copy(mod))
  if(grds is not None):
    grd.append(np.copy(grd))
  if(ress is not None):
    ress.append(np.copy(res))
  if(objs is not None):
    objs.append(np.copy(obj))

class cdstep:
  """ Performs one step of conjugate directions """

  def __init__(self,mshape,dshape):
    """
    Constructor for cdstep

    Parameters:
      mshape - list of shapes of model vectors
      dshape - list of shapes of data vectors
    """
    # Allocate temporary data list
    if(isinstance(dshape,list)):
      self.ss = []
      for dsz in dshape:
        self.ss.append(np.zeros(dsz))
    else:
      self.ss = np.zeros(dshape)

    # Allocate temporary model list
    if(isinstance(mshape,list)):
      self.s = []
      for msz in mshape:
        self.s.append(np.zeros(msz))
    else:
      self.s = np.zeros(mshape)

    # Member variables
    self.small = 1e-20; self.fiter = True
  
  def step(self,m,g,rr,gg) -> None:
    """
    Performs one step of conjugate directions

    Parameters:
      m  - model for the current iteration
      g  - gradient for the current iteration
      rr - residual for the current iteration
      gg - data space gradient for current iteration

    Updates both the m and rr arrays
    """
    # Step sizes
    alfa = None; beta = None

    if(self.fiter):
      # First iteration steepest descent
      beta = 0
      gg_gg = gdot(gg,gg)
      if(np.abs(gg_gg) < self.small):
        print("gg.dot(gg) == %g. Cannot find a proper step size, "
            "will terminate solver"%(gg_gg))
        return False
      # Compute step length
      alfa = -gdot(gg,rr)/gg_gg
      self.fiter = False
    else:
      # Search a plane (invert a 2X2 matrix) for alfa and beta
      gg_gg = gdot(gg,gg)
      ss_ss = gdot(self.ss,self.ss)
      gg_ss = gdot(gg,self.ss)
      if(np.abs(gg_gg) < self.small or np.abs(ss_ss) < self.small):
        print("gg.dot(gg) == 0 or ss.dot(ss) == 0. Cannot find a proper step size, "
            "will terminate solver")
        return False
      determ = gg_gg * ss_ss * (1.0 - gg_ss/gg_gg * gg_ss/ss_ss);
      gg_rr = -gdot(gg,rr)
      ss_rr = -gdot(self.ss,rr)
      # Compute step sizes
      alfa = ( ss_ss * gg_rr - gg_ss * ss_rr)/determ
      beta = (-gg_ss * gg_rr + gg_gg * ss_rr)/determ
      
    # s  = beta*s  + alpha*g
    scale_add(self.s, beta,g, alfa)
    # ss = beta*ss + alpha*gg
    scale_add(self.ss,beta,gg,alfa)

    # m  += s
    scale_add(m, 1.0,self.s, 1.0)
    # rr += ss
    scale_add(rr,1.0,self.ss,1.0)

    return True

def gdot(la,lb):
  """
  Computes the dot product between two vectors.
  Handles the case when la and lb are lists of numpy arrays

  Parameters:
    la - a numpy array or list of numpy arrays
    lb - a numpy array or list of numpy arrays

  Returns the dot product between la and lb
  """
  if(not isinstance(la,list) and not isinstance(lb,list)):
    return np.dot(la.flatten(),lb.flatten())
  else:
    if(len(la) != len(lb)):
      raise Exception("Length of la (%d) must equal length of lb (%d)"%(len(la),len(lb)))
    dd = 0
    for k in range(len(la)):
      dd += np.dot(la[k].flatten(),lb[k].flatten())
    return dd

def swap(a,b):
  """
  Swaps a with b

  Parameters:
    a - input numpy array
    b - input numpy array
  """
  t = np.zeros(a.shape,dtype='float32')
  t[:] = a[:]
  a[:] = b[:]
  b[:] = t[:]

def scale_add(a,sca,b,scb) -> None:
  """
  Computes the following sum
  a = sca*a + b*scb

  Parameters:
    a   - a numpy array or list of numpy arrays
    sca - a scalar scaling the vector a
    b   - a numpy array or list of numpy arrays
    scb - a scalar scaling the vector b
  """
  if(not isinstance(a,list) and not isinstance(b,list)):
    a[:] = sca*a + scb*b
  else:
    if(len(a) != len(b)):
      raise Exception("Length of a (%d) must equal length of b (%d)"%(len(a),len(b)))
    for k in range(len(a)):
      a[k][:] = sca*a[k] + scb*b[k]

