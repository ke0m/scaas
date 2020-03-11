import os,string
import numpy as np
from utils.ptyprint import create_inttag
import inpout.seppy as seppy

class optqc:
  """ Class for QC'ing the optimization while it is running """

  def __init__(self,prefix,nmods=1,nress=1,mkddir=False,trials=False,trpars=None,sep=False,fig=True,**kwargs):
    """ 
    optqc constructor

    Parameters
      prefix - prefix for the name of the directory to be created 
               containing all QC files
      nmods  - number of models which are being estimated [1]
      nress  - number of terms in the total objective function
      mkddir - flag of whether or not to write out the modeled data
               or data residuals [False]
      trials - whether to write out the linesearch trials for nonlinear
               optimization [False]
      trpars - parameters for trimming models and gradients before viewing [None]
    """
    # First make the directory
    odir = prefix + '-' + create_inttag(0,100)
    if(os.path.isdir(odir) == False):
      os.mkdir(odir)
    else:
      num = int(odir.split('-')[-1])
      os.mkdir(prefix + '-' + create_inttag(num,100))

    # Now make the subdirectories
    if(not trials):
      ofdir = odir + kwargs.get('ofdir','/ofn'); mddir = odir + kwargs.get('mddir','/mod'); grdir = odir + kwargs.get('grdir','/grd')
      os.mkdir(ofdir); os.mkdir(mddir); os.mkdir(grdir)
      if(mkddir):
        dtdir = odir + kwargs.get('dtdir','/dat')
        os.mkdir(dtdir)
    # Make the trials if requested
    else:
      ofdir = odir + kwargs.get('ofdir','/ofntrial'); mddir = odir + kwargs.get('mddir','/modtrial'); grdir = odir + kwargs.get('grdir','/grdtrial')
      os.mkdir(ofdir); os.mkdir(mddir); os.mkdir(grdir)
      if(mkddir):
        dtdir = odir + kwargs.get('dtdir','/dattrial')
        os.mkdir(dtdir)

    # File names
    self.ofhfname = kwargs.get('ofhfname',ofdir + '/ofn.H')
    self.mdhfname = kwargs.get('mdhfname',mddir + '/mod.H')
    self.grhfname = kwargs.get('grhfname',grdir + '/grd.H')
    if(mkddir):
      self.dthfname = kwargs.get('dthfname',dtdir + '/dat.H')

    # Figure name prefix
    self.figpfx = kwargs.get('figpfx','iter')

    # Create the axes for figures and sep writing if desired
    self.ofaxes = seppy.axes([1],[0.0],[1.0])
    # Model and gradient axes
    self.nms  = kwargs.get('nms',[])
    self.mdim = len(self.nms)
    self.dms  = kwargs.get('dms',[])
    self.oms  = kwargs.get('oms',[])
    # Create axes if possible
    if(len(self.nms) != 0):
      self.maxes = seppy.axes(nms,oms,dms)
    # Data axes
    self.nds  = kwargs.get('nds',[])
    self.dnim = len(self.nds)
    self.dds  = kwargs.get('dds',[])
    self.ods  = kwargs.get('ods',[])
    # Create axes if possible
    if(len(self.nds) != 0):
      self.daxes = seppy.axes(nds,ods,dds)

    # If SEP writing is desired, setup the IO
    if(sep):
      self.sep = seppy.sep([])

    # Internal iteration counter
    self.iter = 1

    # Save figure parameters
    # General parameters
    if(fig):
      self.hbox      = kwargs.get('hbox',8)
      self.wbox      = kwargs.get('wbox',8)
      self.labelsize = kwargs.get('labelsize',14)
      self.tickparam = kwargs.get('ticksize',14)
      self.title     = kwargs.get('title','Iter=%s')
      self.seethru   = kwargs.get('transparent',True)
      self.dpi       = kwargs.get('dpi',150)
      self.niter     = kwargs.get('niter',10000)
      self.figext    = kwargs.get('figxt','png')
      # Objective function parameters
      self.oflwidth  = kwargs.get('oflwidth',1.5)
      self.oflcolor  = kwargs.get('oflcolor','tab:blue')
      self.ofylabel  = kwargs.get('ofylabel','Loss function')
      self.ofxlabel  = kwargs.get('ofxlabel','Iteration')
      # Model parameters
      self.mdlwidth = kwargs.get('mdlwidth',1.5)
      self.mdlcolor = kwargs.get('mdlcolor','tab:blue')
      self.mdcmap   = kwargs.get('mdcmap','gray')
      self.mdxlabel = kwargs.get('mdxlabel', ' ')
      self.mdylabel = kwargs.get('mdylabel', ' ')
      self.mdvmin   = kwargs.get('mdvmin',None)
      self.mdvmax   = kwargs.get('mdvmax',None)
      self.mtransp  = kwargs.get('mtransp',False)
      self.minterp  = kwargs.get('minterp',None)
      # Gradient parameters
      self.grlwidth = kwargs.get('grlwidth',1.5)
      self.grlcolor = kwargs.get('grlcolor','tab:blue')
      self.grcmap   = kwargs.get('grcmap','gray')
      self.grvmin   = kwargs.get('grvmin',None)
      self.grvmax   = kwargs.get('grvmax',None)
      # Data parameters
      self.dtlwidth = kwargs.get('dtlwidth',1.5)
      self.dtlcolor = kwargs.get('dtlcolor','tab:blue')
      self.dtcmap   = kwargs.get('dtcmap','gray')
      self.dtxlabel = kwargs.get('dtxlabel', ' ')
      self.dtylabel = kwargs.get('dtylabel', ' ')
      self.dtvmin   = kwargs.get('dtvmin',None)
      self.dtvmax   = kwargs.get('dtvmax',None)
      self.dtransp  = kwargs.get('dtransp',None)
      self.dinterp  = kwargs.get('dinterp',None)

  def output(self,ofn,mod,grd,res):
    """ 
    Writes the optimization parameters to files for the current iteration

    Parameters
      ofn - objective function
      mod - model vector
      grd - gradient vector
      res - residual vector
    """
    # Create the axes if not possible in constructor
    if(len(self.nms) == 0):
      if(self.iter == 1):
        self.mdim = len(mod.shape)
        for idim in range(self.mdim):
          self.nms.append(mod.shape[idim])
        if(len(self.dms) == 0):
          self.dms = np.ones(self.mdim)
        if(len(self.oms) == 0):
          self.oms = np.zeros(self.mdim)
      if(sep):
        # Create axes if SEP writing
        self.maxes  = seppy.axes(self.nms,self.oms,self.dms)
    if(self.mkddir):
      if(len(self.nds) == 0):
        if(self.iter == 1):
          self.ddim = len(res[0].shape)
          for idim in range(self.ddim):
            self.nds.append(dat.shape[idim])
        if(len(self.dds) == 0):
          self.dds = np.ones(self.ddim)
        if(len(self.ods) == 0):
          self.ods = np.zeros(self.ddim)
      if(sep):
        # Create axes if SEP writing
        self.daxes  = seppy.axes(self.nds,self.ods,self.dds)

    # Write the figures
    if(fig):
      if(len(self.nms) == 1):
        if(self.iter == 1):
          x = np.linspace(oms[0],oms[0]+(nms[0]-1)*dms[0],nms[0])
        # Save model
        fig = plt.figure(figsize=(self.wbox,self.hbox))
        ax = fig.gca()
        ax.plot(x,mod,color=self.mdlcolor,linewidth=self.mdlwidth)
        ax.set_xlabel(self.mdxlabel,fontsize=self.labelsize)
        ax.set_ylabel(self.mdylabel,fontsize=self.labelsize)
        ax.tick_params(labelsize=self.ticksize)
        ax.set_title(self.title%(create_inttag(self.iter,self.niter)),fontsize=self.labelsize)
        plt.savefig(self.mddir + '/' + self.figpfx + self.figext,bbox_inches='tight',dpi=self.dpi,transparent=self.seethru)
        plt.close()
        # Save gradient
        fig = plt.figure(figsize=(self.wbox,self.hbox))
        ax = fig.gca()
        ax.plot(x,grd,color=self.grlcolor,linewidth=self.grlwidth)
        ax.set_xlabel(self.grxlabel,fontsize=self.labelsize)
        ax.set_ylabel(self.grylabel,fontsize=self.labelsize)
        ax.tick_params(labelsize=self.ticksize)
        ax.set_title(self.title%(create_inttag(self.iter,self.niter)),fontsize=self.labelsize)
        plt.savefig(self.grdir + '/' + self.figpfx + self.figext,bbox_inches='tight',dpi=self.dpi,transparent=self.seethru)
        plt.close()
        # Save data
        if(self.mkddir):
          fig = plt.figure(figsize=(self.wbox,self.hbox))
          ax = fig.gca()
          ax.plot(x,res[0],color=self.dtlcolor,linewidth=self.dtlwidth)
          ax.set_xlabel(self.dtxlabel,fontsize=self.labelsize)
          ax.set_ylabel(self.dtylabel,fontsize=self.labelsize)
          ax.tick_params(labelsize=self.ticksize)
          ax.set_title(self.title%(create_inttag(self.iter,self.niter)),fontsize=self.labelsize)
          plt.savefig(self.dtdir + '/' + self.figpfx + self.figext,bbox_inches='tight',dpi=self.dpi,transparent=self.seethru)
          plt.close()
      elif(len(self.nms) == 2):
        # Save model
        fig = plt.figure(figsize=(self.wbox,self.hbox))
        ax = fig.gca()
        if(self.mdvmin == None or self.mdvmax == None):
          self.mdvmin = np.max(mod); self.mdvmax = np.min(mod)
        if(self.mtransp):
          im = ax.imshow(mod.T,color=self.mdcmap,extent=[om[1],om[1]+(nm[1]-1)*dm[1],om[0],om[0]+(nm[0]-1)*dm[0]],vmin=self.mdvmin,vmax=self.mdvmax)
        else:
          im = ax.imshow(mod,color=self.mdcmap,extent=[om[1],om[1]+(nm[1]-1)*dm[1],om[0],om[0]+(nm[0]-1)*dm[0]],vmin=self.mdvmin,vmax=self.mdvmax)
        ax.set_xlabel(self.mdxlabel,fontsize=self.labelsize)
        ax.set_ylabel(self.mdylabel,fontsize=self.labelsize)
        ax.tick_params(labelsize=self.ticksize)
        ax.set_title(self.title%(create_inttag(self.iter,self.niter)),fontsize=self.labelsize)
        plt.savefig(self.mddir + '/' + self.figpfx + self.figext,bbox_inches='tight',dpi=self.dpi,transparent=self.seethru)
        plt.close()
        # Save gradient
        fig = plt.figure(figsize=(self.wbox,self.hbox))
        ax = fig.gca()
        if(self.grvmin == None or self.grvmax == None):
          self.mdvmin = np.max(mod); self.mdvmax = np.min(mod)
        if(self.mtransp):
          im = ax.imshow(grd.T,color=self.mdcmap,extent=[om[1],om[1]+(nm[1]-1)*dm[1],om[0],om[0]+(nm[0]-1)*dm[0]],vmin=self.mdvmin,vmax=self.mdvmax)
        else:
          im = ax.imshow(grd,color=self.mdcmap,extent=[om[1],om[1]+(nm[1]-1)*dm[1],om[0],om[0]+(nm[0]-1)*dm[0]],vmin=self.mdvmin,vmax=self.mdvmax)
        ax.set_xlabel(self.mdxlabel,fontsize=self.labelsize)
        ax.set_ylabel(self.mdylabel,fontsize=self.labelsize)
        ax.tick_params(labelsize=self.ticksize)
        ax.set_title(self.title%(create_inttag(self.iter,self.niter)),fontsize=self.labelsize)
        plt.savefig(self.grdir + '/' + self.figpfx + self.figext,bbox_inches='tight',dpi=self.dpi,transparent=self.seethru)
        plt.close()
        # Save data
        if(self.mkddir):
          fig = plt.figure(figsize=(self.wbox,self.hbox))
          ax = fig.gca()
          if(self.dtvmin == None or self.dtvmax == None):
            self.dtvmin = np.max(res[0]); self.dtvmax = np.min(res[0])
          if(self.dtransp):
            im = ax.imshow(res[0].T,color=self.dtcmap,extent=[od[1],od[1]+(nd[1]-1)*dd[1],od[0],od[0]+(nd[0]-1)*dd[0]],vmin=self.dtvmin,vmax=self.dtvmax)
          else:
            im = ax.imshow(res[0],color=self.dtcmap,extent=[od[1],od[1]+(nd[1]-1)*dd[1],od[0],od[0]+(nd[0]-1)*dd[0]],vmin=self.dtvmin,vmax=self.dtvmax)
          ax.set_xlabel(self.dtxlabel,fontsize=self.labelsize)
          ax.set_ylabel(self.dtylabel,fontsize=self.labelsize)
          ax.tick_params(labelsize=self.ticksize)
          ax.set_title(self.title%(create_inttag(self.iter,self.niter)),fontsize=self.labelsize)
          plt.savefig(self.dtdir + '/' + self.figpfx + self.figext,bbox_inches='tight',dpi=self.dpi,transparent=self.seethru)
          plt.close()
      
    # Write the SEP files
    if(sep):
      if(self.iter == 1):
        self.sep.write_file(None, self.ofaxes, ofn, ofname=self.ofhfname)
        # Make ofaxes zero-dimensional (for append_to_movie)
        self.ofaxes.ndims = 0
        self.sep.write_file(None, self.maxes, mod, ofname=self.mdhfname)
        self.sep.write_file(None, self.maxes, grd, ofname=self.grhfnmae)
        if(self.mkddir):
          self.sep.write_file(None, self.daxes, res[0], ofname=self.dthfname)
      # Append to files for subsequent iterations
      else:
        self.sep.append_to_movie(None, self.ofaxes, ofn, self.iter, ofname=self.ohfname)
        self.sep.append_to_movie(None, self.maxes, mod, self.iter, ofname=self.mdhfname)
        self.sep.append_to_movie(None, self.maxes, grd, self.iter, ofname=self.grhfname)
        if(self.mkddir):
          self.sep.append_to_movie(None, self.daxes, res, self.iter, ofname=self.dthfname)
    # Update the iteration counter
    self.iter += 1

#class optqc:
#  """ Optimization QC """
#  def __init__(self,sep,oftag,ofaxes,mtag,gtag,mgaxes,dtag=None,daxes=None,trials=False,trpars=None,figpars=None):
#    # Iteration counter
#    self.iter = 1
#    # Tags and axes
#    self.oftag = oftag; self.ofaxes = ofaxes                     # Objective fucntion
#    self.mtag  = mtag;  self.gtag   = gtag; self.mgaxes = mgaxes # Model and gradient
#    self.dtag  = dtag;  self.daxes = daxes                       # Data
#    # IO object
#    self.sep = sep
#    # Check if the files exist and set flags
#    self.ofout = False; self.mout = False; self.gout = False; self.dout = False
#    if(self.sep.get_fname(self.oftag) != None):
#      self.ofout = True
#    if(self.sep.get_fname(self.mtag) != None):
#      self.mout = True
#    if(self.sep.get_fname(self.gtag) != None):
#      self.gout = True
#    if(self.sep.get_fname(self.dtag) != None):
#      self.dout = True
#    # Create the trial file names if requested
#    self.trials = trials
#    if(self.trials):
#      # Initialize output files
#      self.oftrial = None; self.mtrial = None; self.gtrial = None; self.dtrial = None
#      # Objective function
#      if(self.sep.get_fname(self.oftag) != None):
#        ofname = self.sep.get_fname(self.oftag)
#        self.oftrial = ofname.split('.')[0] + '-trial.H'
#      # Model
#      if(self.sep.get_fname(self.mtag) != None):
#        mname = self.sep.get_fname(self.mtag)
#        self.mtrial = mname.split('.')[0] + '-trial.H'
#      # Gradient
#      if(self.sep.get_fname(self.gtag) != None):
#        gname = self.sep.get_fname(self.gtag)
#        self.gtrial = gname.split('.')[0] + '-trial.H'
#      # Data
#      if(self.sep.get_fname(self.dtag) != None):
#        dname = self.sep.get_fname(self.dtag)
#        self.dtrial = dname.split('.')[0] + '-trial.H'
#    # Parameters for trimming the padding before writing
#    self.trmog = False; self.trdat = False
#    if(trpars != None):
#      if(self.mout != False or self.gout != False):
#        # Left/right Top/bottom indices for unpadding the model
#        self.lidx = trpars['lidx']; self.ridx = trpars['ridx']
#        self.tidx = trpars['tidx']; self.bidx = trpars['bidx']
#        self.trmog = True
#
#  def outputH(self,ofn,mod,grd,dat=None):
#    """ Main output function for writing the iterates to SEPlib files """
#    if(self.trials):
#      self.output_trial(ofn,mod,grd,dat)
#    else:
#      self.output_iter(ofn,mod,grd,dat)
#
#  def output_trial(self,ofn,mod,grd,dat=None):
#    """ Writes the trial results to files """
#    # Write the file if the first iteration
#    if(self.iter == 1):
#      if(self.oftrial != None):
#        self.sep.write_file(None, self.ofaxes, ofn, ofname=self.oftrial)
#        # Make ofaxes zero-dimensional (for append_to_movie)
#        self.ofaxes.ndims = 0
#      if(self.mtrial != None):
#        self.sep.write_file(None, self.mgaxes, self.trim_mog(mod), ofname=self.mtrial)
#      if (self.gtrial != None):
#        self.sep.write_file(None, self.mgaxes, self.trim_mog(grd), ofname=self.gtrial)
#      if(self.dtrial != None):
#        self.sep.write_file(None, self.daxes, dat, ofname=self.dtrial)
#    # Append to files for subsequent iterations
#    else:
#      if(self.oftrial != None):
#        self.sep.append_to_movie(None, self.ofaxes, ofn, self.iter, ofname=self.oftrial)
#      if(self.mtrial != None):
#        self.sep.append_to_movie(None, self.mgaxes, self.trim_mog(mod), self.iter, ofname=self.mtrial)
#      if(self.gtrial != None):
#        self.sep.append_to_movie(None, self.mgaxes, self.trim_mog(grd), self.iter, ofname=self.gtrial)
#      if(self.dtrial != None):
#        self.sep.append_to_movie(None, self.daxes, dat, self.iter, ofname=self.dtrial)
#    # Update the iteration counter
#    self.iter += 1
#
#  def output_iter(self,ofn,mod,grd,dat=None):
#    """ Writes the iterates to file (at each iteration) """
#    # Write the file if the first iteration
#    if(self.iter == 1):
#      if(self.ofout):
#        self.sep.write_file(self.oftag,self.ofaxes,ofn)
#        # Make ofaxes zero-dimensional (for append_to_movie)
#        self.ofaxes.ndims = 0
#      if(self.mout):
#        self.sep.write_file(self.mtag, self.mgaxes, self.trim_mog(mod))
#      if(self.gout):
#        self.sep.write_file(self.gtag, self.mgaxes, self.trim_mog(grd))
#      if(self.dout):
#        self.sep.write_file(self.dtag, self.daxes, dat)
#    # Append to files for subsequent iterations
#    else:
#      if(self.ofout):
#        self.sep.append_to_movie(self.oftag, self.ofaxes, ofn, self.iter)
#      if(self.mout):
#        self.sep.append_to_movie(self.mtag, self.mgaxes, self.trim_mog(mod), self.iter)
#      if(self.gout):
#        self.sep.append_to_movie(self.gtag, self.mgaxes, self.trim_mog(grd), self.iter)
#      if(self.dout):
#        self.sep.append_to_movie(self.dtag, self.daxes, dat, self.iter)
#    # Update the iteration counter
#    self.iter += 1
#
#  def trim_mog(self,mog):
#    """ Trims the model or the gradient """
#    if(self.trmog):
#      return mog[self.tidx:self.bidx,self.lidx:self.ridx]
#    else:
#      return mog

