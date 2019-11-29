import numpy as np
import opt.opr8tr as opr

#TODO: 
# 1. cgstep (Nocedal 5.2)
# 2. idcgstep (Nocedal 7.1)
# 3. CD class
#    needs an internal cdstep
#    uses the abstract operator class
# 4. CG class
#    needs an internal cgstep
#    uses the abstract operator class

def cdstep(f,m,g,rr,gg,w,idict) -> None:
  """  Performs one step of conjugate directions as 
  described in GIEE by Claerbout

  Parameters
  ----------
  f    : the objective function value at the current iteration
  m    : the model at the current iteration
  g    : the gradient at the current iteration
  rr   : the residual at the current iteration
  gg   : the data space gradient at the current iteration (conjugate gradient)
  w    : a working dictionary that stores the current previous search direction
  idict: an input dictionary containing information about the current state of the solver

  Returns
  -------
  Does not return anything to be consistent with how LBFGS works. The vectors
  m and rr are updated internally. Also, the elements of the dictionary idict
  are updated internally if the dictionary is passed
  """
  # Tolerance
  small = 1e-20
  # Step sizes
  alpha = None; beta = None;

  ## Preliminaries
  # Check if objective function was reduced
  if(idict['iter'] != 0):
    if(f >= idict['f']):
      print("Did not reduce objective function %f. Exiting"%(f))
      idict['iflag'] = -1
      return

  # Check if tolerance has been reached
  if(f < idict['rtol']):
    print("Tolerance of %f has been reached (f=%f). Exiting"%(idict['rtol'],f))
    idict['iflag'] = 0
    return

  # Print info if desired
  if(idict['verb'] == 1):
    print("iter=%d f=%f gnorm=%f"%(idict['iter'],f,np.dot(g,g)))

  # Get the search directions
  s = w['s']; ss = w['ss']

  # Code for CD step
  if(idict['iter'] == 0):
    # First iteration: steepest descent
    beta = 0
    gg_gg = np.dot(gg,gg)
    if(np.abs(gg_gg) < small):
      print("gg.dot(gg) == %g. Cannot find a proper step size, "
          " will terminate solver"%(gg_gg))
      idict['iflag'] = -1
      return
    # Compute step length
    alfa = -np.dot(gg,rr)/gg_gg
  else:
    # Search a plane (invert a 2X2 matrix) for alfa and beta
    gg_gg = np.dot(gg,gg)
    ss_ss = np.dot(ss,ss)
    gg_ss = np.dot(gg,ss)
    if(np.abs(gg_gg) < small or np.abs(ss_ss) < small):
      print("gg.dot(gg) == 0 or ss.dot(ss) == 0. Cannot find a proper step size, "
          " will terminate solver")
      ifdict['iflag'] = -1
      return
    determ = gg_gg * ss_ss * (1.0 - gg_ss/gg_gg * gg_ss/ss_ss);
    gg_rr = -np.dot(gg,rr)
    ss_rr = -np.dot(ss,rr)
    # Compute step sizes
    alfa = ( ss_ss * gg_rr - gg_ss * ss_rr)/determ
    beta = (-gg_ss * gg_rr + gg_gg * ss_rr)/determ

  # Update the solution and residual steps
  s  = beta*s  + alfa*g
  ss = beta*ss + alfa*gg

  # Update the model and the residual
  m  += s
  rr += ss

  # Update the working dictionary
  w['s']  = s
  w['ss'] = ss

  # Update the info dictionary
  if('alfa' in idict and 'beta' in idict):
    idict['alfa'] = alfa; idict['beta'] = beta
  idict['iter'] += 1
  idict['f'] = f

  if(idict['iter'] > idict['niter']):
    print("Reached maximum number of iterations. Exiting")
    idict['iflag'] = 0
    return

