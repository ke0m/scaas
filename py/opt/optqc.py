
class optqc:
  """ Optimization QC """
  def __init__(self,sep,oftag,ofaxes,mtag,gtag,mgaxes,dtag=None,daxes=None,trials=False):
    # Iteration counter
    self.iter = 1
    # Tags and axes
    self.oftag = oftag; self.ofaxes = ofaxes                     # Objective fucntion
    self.mtag  = mtag;  self.gtag   = gtag; self.mgaxes = mgaxes # Model and gradient
    self.dtag  = dtag;  self.daxes = daxes                       # Data
    # IO object
    self.sep = sep
    # Create the trial file names if requested
    self.trials = trials
    if(self.trials):
      # Initialize output files
      self.oftrial = None; self.mtrial = None; self.gtrial = None; self.dtrial = None
      # Objective function
      if(self.oftag != None):
        ofname = self.sep.get_fname(self.oftag)
        self.oftrial = ofname.split('.')[0] + '-trial.H'
      # Model
      if(self.mtag != None):
        mname = self.sep.get_fname(self.mtag)
        self.mtrial = mname.split('.')[0] + '-trial.H'
      # Gradient
      if(self.gtag != None):
        gname = self.sep.get_fname(self.gtag)
        self.gtrial = gname.split('.')[0] + '-trial.H'
      # Data
      if(self.dtag != None):
        dname = self.sep.get_fname(self.dtag)
        self.dtrial = dname.split('.')[0] + '-trial.H'

  def output(self,ofn,mod,grd,dat=None):
    if(self.trials):
      self.output_trial(ofn,mod,grd,dat)
    else:
      self.output_iter(ofn,mod,grd,dat)

  def output_trial(self,ofn,mod,grd,dat=None):
    # Write the file if the first iteration
    if(self.iter == 1):
      if(self.oftrial != None):
        self.sep.write_file(None, self.ofaxes, ofn, ofname=self.oftrial)
        # Make ofaxes zero-dimensional (for append_to_movie)
        self.ofaxes.ndims = 0
      if(self.mtrial != None):
        self.sep.write_file(None, self.mgaxes, mod, ofname=self.mtrial)
      if (self.gtrial != None):
        self.sep.write_file(None, self.mgaxes, grd, ofname=self.gtrial)
      if(self.dtrial != None):
        self.sep.write_file(None, self.daxes, dat, ofname=self.dtrial)
    # Append to files for subsequent iterations 
    else:
      if(self.oftrial != None):
        self.sep.append_to_movie(None, self.ofaxes, ofn, self.iter, ofname=self.oftrial)
      if(self.mtrial != None):
        self.sep.append_to_movie(None, self.mgaxes, mod, self.iter, ofname=self.mtrial)
      if(self.gtrial != None):
        self.sep.append_to_movie(None, self.mgaxes, grd, self.iter, ofname=self.gtrial)
      if(self.dtrial != None):
        self.sep.append_to_movie(None, self.daxes, dat, self.iter, ofname=self.dtrial)
    # Update the iteration counter
    self.iter += 1

  def output_iter(self,ofn,mod,grd,dat=None):
    # Write the file if the first iteration
    if(self.iter == 1):
      if(self.oftag != None):
        self.sep.write_file(self.oftag,self.ofaxes,ofn)
        # Make ofaxes zero-dimensional (for append_to_movie)
        self.ofaxes.ndims = 0
      if(self.mtag != None):
        self.sep.write_file(self.mtag, self.mgaxes, mod)
      if (self.gtag != None):
        self.sep.write_file(self.gtag, self.mgaxes, grd)
      if(self.dtag != None):
        self.sep.write_file(self.dtag, self.daxes, dat)
    # Append to files for subsequent iterations 
    else:
      if(self.oftag != None):
        self.sep.append_to_movie(self.oftag, self.ofaxes, ofn, self.iter)
      if(self.mtag != None):
        self.sep.append_to_movie(self.mtag, self.mgaxes, mod, self.iter)
      if(self.gtag != None):
        self.sep.append_to_movie(self.gtag, self.mgaxes, grd, self.iter)
      if(self.dtag != None):
        self.sep.append_to_movie(self.dtag, self.daxes, dat, self.iter)
    # Update the iteration counter
    self.iter += 1

