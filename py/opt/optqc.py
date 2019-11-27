
class optqc:
  """ Optimization QC """
  def __init__(self,sep,oftag,ofaxes,mtag,gtag,mgaxes,dtag=None,daxes=None,trials=False,trpars=None,figpars=None):
    # Iteration counter
    self.iter = 1
    # Tags and axes
    self.oftag = oftag; self.ofaxes = ofaxes                     # Objective fucntion
    self.mtag  = mtag;  self.gtag   = gtag; self.mgaxes = mgaxes # Model and gradient
    self.dtag  = dtag;  self.daxes = daxes                       # Data
    # IO object
    self.sep = sep
    # Check if the files exist and set flags
    self.ofout = False; self.mout = False; self.gout = False; self.dout = False
    if(self.sep.get_fname(self.oftag) != None):
      self.ofout = True
    if(self.sep.get_fname(self.mtag) != None):
      self.mout = True
    if(self.sep.get_fname(self.gtag) != None):
      self.gout = True
    if(self.sep.get_fname(self.dtag) != None):
      self.dout = True
    # Create the trial file names if requested
    self.trials = trials
    if(self.trials):
      # Initialize output files
      self.oftrial = None; self.mtrial = None; self.gtrial = None; self.dtrial = None
      # Objective function
      if(self.sep.get_fname(self.oftag) != None):
        ofname = self.sep.get_fname(self.oftag)
        self.oftrial = ofname.split('.')[0] + '-trial.H'
      # Model
      if(self.sep.get_fname(self.mtag) != None):
        mname = self.sep.get_fname(self.mtag)
        self.mtrial = mname.split('.')[0] + '-trial.H'
      # Gradient
      if(self.sep.get_fname(self.gtag) != None):
        gname = self.sep.get_fname(self.gtag)
        self.gtrial = gname.split('.')[0] + '-trial.H'
      # Data
      if(self.sep.get_fname(self.dtag) != None):
        dname = self.sep.get_fname(self.dtag)
        self.dtrial = dname.split('.')[0] + '-trial.H'
    # Parameters for trimming the padding before writing
    self.trmog = False; self.trdat = False
    if(trpars != None):
      if(self.mout != False or self.gout != False):
        # Left/right Top/bottom indices for unpadding the model
        self.lidx = trpars['lidx']; self.ridx = trpars['ridx']
        self.tidx = trpars['tidx']; self.bidx = trpars['bidx']
        self.trmog = True

  def outputH(self,ofn,mod,grd,dat=None):
    """ Main output function for writing the iterates to SEPlib files """
    if(self.trials):
      self.output_trial(ofn,mod,grd,dat)
    else:
      self.output_iter(ofn,mod,grd,dat)

  def output_trial(self,ofn,mod,grd,dat=None):
    """ Writes the trial results to files """
    # Write the file if the first iteration
    if(self.iter == 1):
      if(self.oftrial != None):
        self.sep.write_file(None, self.ofaxes, ofn, ofname=self.oftrial)
        # Make ofaxes zero-dimensional (for append_to_movie)
        self.ofaxes.ndims = 0
      if(self.mtrial != None):
        self.sep.write_file(None, self.mgaxes, self.trim_mog(mod), ofname=self.mtrial)
      if (self.gtrial != None):
        self.sep.write_file(None, self.mgaxes, self.trim_mog(grd), ofname=self.gtrial)
      if(self.dtrial != None):
        self.sep.write_file(None, self.daxes, dat, ofname=self.dtrial)
    # Append to files for subsequent iterations
    else:
      if(self.oftrial != None):
        self.sep.append_to_movie(None, self.ofaxes, ofn, self.iter, ofname=self.oftrial)
      if(self.mtrial != None):
        self.sep.append_to_movie(None, self.mgaxes, self.trim_mog(mod), self.iter, ofname=self.mtrial)
      if(self.gtrial != None):
        self.sep.append_to_movie(None, self.mgaxes, self.trim_mog(grd), self.iter, ofname=self.gtrial)
      if(self.dtrial != None):
        self.sep.append_to_movie(None, self.daxes, dat, self.iter, ofname=self.dtrial)
    # Update the iteration counter
    self.iter += 1

  def output_iter(self,ofn,mod,grd,dat=None):
    """ Writes the iterates to file (at each iteration) """
    # Write the file if the first iteration
    if(self.iter == 1):
      if(self.ofout):
        self.sep.write_file(self.oftag,self.ofaxes,ofn)
        # Make ofaxes zero-dimensional (for append_to_movie)
        self.ofaxes.ndims = 0
      if(self.mout):
        self.sep.write_file(self.mtag, self.mgaxes, self.trim_mog(mod))
      if(self.gout):
        self.sep.write_file(self.gtag, self.mgaxes, self.trim_mog(grd))
      if(self.dout):
        self.sep.write_file(self.dtag, self.daxes, dat)
    # Append to files for subsequent iterations
    else:
      if(self.ofout):
        self.sep.append_to_movie(self.oftag, self.ofaxes, ofn, self.iter)
      if(self.mout):
        self.sep.append_to_movie(self.mtag, self.mgaxes, self.trim_mog(mod), self.iter)
      if(self.gout):
        self.sep.append_to_movie(self.gtag, self.mgaxes, self.trim_mog(grd), self.iter)
      if(self.dout):
        self.sep.append_to_movie(self.dtag, self.daxes, dat, self.iter)
    # Update the iteration counter
    self.iter += 1

  def trim_mog(self,mog):
    """ Trims the model or the gradient """
    if(self.trmog):
      return mog[self.tidx:self.bidx,self.lidx:self.ridx]
    else:
      return mog

