import numpy as np
import os as opsys
import socket,getpass
import __main__ as main
import datetime
import string, random
import subprocess as sp
import matplotlib.pyplot as plt

class axes:
  """ Axes of regularly sampled data"""
  def __init__(self,n,o,d,label=None):
    self.ndims = len(n)
    assert(len(o) == self.ndims or len(d) == self.ndims), "Error: the size of n, o and d do not match."
    self.n = n
    self.o = o
    self.d = d
    self.label = label

  def get_nelem(self):
    return np.prod(self.n)

  def add_axis(self,nin,oin,din):
    self.n.append(nin);
    self.o.append(oin);
    self.d.append(din);
    self.ndims += 1

#TODO: Need to be able to read and write complex data
class sep:
  """ Utility for reading, writing and plotting SEP files """

  def __init__(self,argv=None):
    self.hdict = {}
    self.haxes = {}
    self.argv  = argv
    self.hostname = self.gethostname()

  def get_fname(self, tag):
    """ Gets the file name with the associated tag """
    if(self.argv is None):
      raise Exception("Must pass sys.argv to constructor in order to use tags")
    fname = None
    for iarg in self.argv:
      keyval = iarg.split('=')
      if(len(keyval) != 2): continue
      if(keyval[0] == tag):
        fname = keyval[1]
        break

    return fname

  def read_header(self,ifname,tag=None):
    """ Reads a SEP header file from tag and returns the axes """
    self.hdict = {}
    # Get the filename with the given tag
    assert(tag != None or ifname != None), "Need either a tag or inputfile for reading.\n"
    if(tag != None):
      hin  = self.get_fname(tag)
      assert (hin != None), "No file associated with tag '%s'"%(tag)
    else:
      hin = ifname
    # Work with tags if argv has been passed
    if(self.argv is not None):
      hout = self.get_fname("out")
      fout = None
      if(hout != None):
        fout = open(hout,"w")
    # First read header into dictionary
    for line in open(hin).readlines():
      # If output file, copy history to output
      if(tag == "in" and hout != None):
        fout.write(line)
      splitspace = line.split(' ')
      for item in splitspace:
        spliteq = item.split('=')
        if(len(spliteq) == 1): continue
        spliteq[0] = spliteq[0].replace('\n','')
        spliteq[0] = spliteq[0].replace('\t','')
        spliteq[1] = spliteq[1].replace('\n','')
        spliteq[1] = spliteq[1].replace('\t','')
        self.hdict[spliteq[0]] = spliteq[1]
    # Check if it found a binary
    assert("in" in self.hdict), "Error: header in file %s does not have an associated binary."%(hin)
    self.hdict["in"] = self.hdict["in"].replace('"','')
    # Read the header info into a list of axes
    ns = []; os = []; ds = []; lbls = []
    for n in range(1,7):
      nkey   = "n" + str(n)
      okey   = "o" + str(n)
      dkey   = "d" + str(n)
      lblkey = "label" + str(n)
      if n == 1:
        assert (nkey in self.hdict), "Error: header in file %s has no n1."%(hin)
      if nkey in self.hdict and okey in self.hdict and dkey in self.hdict:
        ns.append(int(self.hdict[nkey]))
        os.append(float(self.hdict[okey]))
        ds.append(float(self.hdict[dkey]))
        if(lblkey in self.hdict):
          lbls.append(self.hdict[lblkey])
        else:
          lbls.append(" ")

    # Remove ones at the end
    for n in ns:
      if(ns[-1] == 1.0):
        del ns[-1]; del os[-1]; del ds[-1]; del lbls[-1]

    # Take care of the remaining
    if ns[-1] == 1.0:
        del ns[-1]; del os[-1]; del ds[-1]; del lbls[-1]

    if(lbls == []):
      return axes(ns,os,ds,None)
    else:
      return axes(ns,os,ds,lbls)

  def read_header_dict(self,ifname,tag=None):
    """ Reads a SEP header file and returns a dictionary """
    hdict = {}
    # Get the filename with the given tag
    assert(tag != None or ifname != None), "Need either a tag or inputfile for reading.\n"
    if(tag != None):
      hin  = self.get_fname(tag=tag)
      assert (hin != None), "No file associated with tag '%s'"%(tag)
    else:
      hin = ifname
    if(self.argv is not None):
      hout = self.get_fname(tag="out")
    else:
      hout = None
    fout = None
    if(hout != None):
      fout = open(hout,"w")
    # Read header into dictionary
    for line in open(hin).readlines():
      splitspace = line.split(' ')
      for item in splitspace:
        spliteq = item.split('=')
        if(len(spliteq) == 1): continue
        spliteq[0] = spliteq[0].replace('\n','')
        spliteq[0] = spliteq[0].replace('\t','')
        spliteq[1] = spliteq[1].replace('\n','')
        spliteq[1] = spliteq[1].replace('\t','')
        hdict[spliteq[0]] = spliteq[1]

    return hdict

  def read_file(self,ifname,form='xdr',safe=False,tag=None):
    """ Reads a SEP file from tag and returns the data and the axes """
    faxes  = self.read_header(ifname,tag)
    if(safe):
      dat = np.zeros(faxes.get_nelem())
      with open(self.hdict["in"],'rb') as f:
        if(form == 'xdr'):
          dat[:] = np.fromfile(f, dtype='>f')
        elif(form == 'native'):
          dat[:] = np.fromfile(f, dtype='<f')
        else:
          print("Failed to read in file. Format %s not recognized\n"%(form))
    else:
      with open(self.hdict["in"],'rb') as f:
        if(form == 'xdr'):
          dat = np.fromfile(f, dtype='>f')
        elif(form == 'native'):
          dat = np.fromfile(f, dtype='<f')
        else:
          print("Failed to read in file. Format %s not recognized\n"%(form))

    return faxes, dat

  def from_header(self,ifname,keys,tag=None):
    """ Given a list of keys (strings), returns the values from the header """
    odict= {}
    # Read the header dictionary
    thdict = self.read_header_dict(ifname,tag)
    # Loop over all keys
    for ikey in keys:
      if ikey in thdict:
        odict[ikey] = thdict[ikey]

    return odict

  def write_header(self,ofname,ns,os=None,ds=None,ofaxes=None,tag=None,dpath=None,form='xdr'):
    """ 
    Writes header information to SEP file and returns
    the path to the output 
    """
    fout = None
    assert(tag != None or ofname != None), "Need a tag or output file name to write a header."
    if(tag != None):
      ofname = self.get_fname(tag)

    if(tag == "out"):
      assert(ofname != None), "No output file name found. To work with tags you must pass argv to constructor."
      fout = open(ofname,"a")
    else:
      fout = open(ofname,"w+")
    # Write the first line
    fout.write('\n' + self.get_fline()+'\n')
    # Get the datapath
    if(len(ofname.split('/')) > 1):
      ofname = ofname.split('/')[-1]
    opath = None
    if(dpath == None):
      opath = self.get_datapath() + ofname + "@"
    else:
      opath = dpath + ofname + "@"
    fout.write('\t\tsets next: in="%s"\n'%(opath))
    if(ofaxes is not None):
      # Print axes
      for k in range(ofaxes.ndims):
        if(ofaxes.label != None):
          fout.write("\t\tn%d=%d o%d=%f d%d=%.12f label%d=%s\n"%
              (k+1,ofaxes.n[k],k+1,ofaxes.o[k],k+1,ofaxes.d[k],k+1,ofaxes.label[k]))
        else:
          fout.write("\t\tn%d=%d o%d=%f d%d=%.12f\n"%
              (k+1,ofaxes.n[k],k+1,ofaxes.o[k],k+1,ofaxes.d[k]))
    else:
      # Write with as much info as they provide
      ndim = len(ns)
      if(os is None):
        os = np.zeros(ndim)
      if(ds is None):
        ds = np.ones(ndim)
      else:
        if(len(ds) != ndim):
          ds = np.pad(ds,(0,ndim-len(ds)),mode='constant',constant_values=1)
        if(len(os) != ndim):
          os = np.pad(os,(0,ndim-len(os)),mode='constant',constant_values=0.0)
      for k in range(ndim):
        fout.write("\t\tn%d=%d o%d=%f d%d=%.12f\n"%
            (k+1,ns[k],k+1,os[k],k+1,ds[k]))

    if(form == 'xdr'):
      fout.write('\t\tdata_format="xdr_float" esize=4\n')
    elif(form == 'native'):
      fout.write('\t\tdata_format="native_float" esize=4\n')
    else:
      print("Error: format %s not recognized"%(form))

    fout.close()

    return opath

  def write_file(self,ofname,data,os=None,ds=None,ofaxes=None,tag=None,dpath=None,form='xdr'):
    """ Writes data and axes to a SEP header and binary """
    opath = self.write_header(ofname,data.shape,os,ds,ofaxes,tag,dpath,form)
    with open(opath,'wb') as f:
      if(form == 'xdr'):
        data.flatten('F').astype('>f').tofile(f)
      elif(form == 'native'):
        data.flatten('F').astype('<f').tofile(f)
      else:
        print("Failed to write file. Format %s not recognized\n"%(form))

  def to_header(self,ofname,info,tag=None):
    """ Writes any auxiliary information to header """
    fout = None
    assert(tag != None or ofname != None), "Need a tag or output file name to write a header."
    if(tag != None):
      ofname = self.get_fname(tag)

    # Open file header
    if(tag == "out"):
      assert(ofname != None), "No output file name found. To work with tags, you must pass argv to constructor."
      fout = open(ofname,"a")
    else:
      fout = open(ofname,"a+")
    fout.write('\n' + info)
    fout.close()

  def write_dummyaxis(self,ofname,dim,tag=None):
    """ Writes a single axis to a SEP header """
    if(ofname == None):
      ofname = self.get_fname(tag)

    fout = open(ofname,"a")
    fout.write("\n\t\tn%d=1 o%d=0.0 d%d=1.0\n"%(dim,dim,dim))
    fout.close()

  def append_to_movie(self,ofname,ofaxes,data,niter,tag=None,dpath=None,form='xdr'):
    """ Appends to a file for an inversion movie"""
    if(ofname == None):
      ofname = self.get_fname(tag)

    # Write an extra line to the header
    fout = open(ofname,"a")
    odim = ofaxes.ndims + 1
    fout.write("\n\t\tn%d=%d o%d=0.0 d%d=1.0\n"%(odim,niter,odim,odim))
    fout.close()

    # Append the data to the binary
    opath = None
    if(dpath == None):
      opath = self.get_datapath() + ofname + "@"
    else:
      opath = dpath + ofname + "@"
    with open(opath,'ab') as f:
      if(form == 'xdr'):
        data.flatten('F').astype('>f').tofile(f)
      elif(form == 'native'):
        data.flatten('F').astype('<f').tofile(f)

  def get_fline(self):
    """ Returns the first line of the program header """
    if(self.argv is None):
      fline = "%s"%(main.__file__)
    else:
      fline = self.argv[0]
    # Get user and hostname
    username = getpass.getuser()
    fline += ":\t" + username + "@" + self.hostname + "\t\t"
    # Get time and date
    time = datetime.datetime.today()
    fline += time.strftime("%a %b %d %H:%M:%S %Y")

    return fline

  def get_datapath(self):
    """ Gets the set datpath for writing SEP binaries """
    dpath = None
    # Check if passed as argument
    if(self.argv is not None):
      dpath = self.get_fname("datapath")
    if(dpath == None):
      # Look in home directory
      datstring = opsys.environ['HOME'] + "/.datapath"
      if(opsys.path.exists(datstring) == True):
        nohost = ''
        # Assumes structure as host datapath=path
        for line in open(datstring).readlines():
          hostpath = line.split()
          if(len(hostpath) == 1):
            nohost = hostpath
          elif(len(hostpath) == 2):
            # Check if hostname matches
            if(self.hostname == hostpath[0]):
              dpath = hostpath[1].split('=')[1]
              break
        if(dpath == None and nohost != None):
          dpath = nohost[0].split('=')[1]
    # Lastly, look at environment variable
    elif(dpath == None and "DATAPATH" in opsys.environ):
      dpath = opsys.environ['DATAPATH']
    #else:
    #  dpath = '/tmp/'

    return dpath

  def gethostname(alias=True):
    """
    An extension of socket.gethostname() where the hostname alias
    is returned if requested. This makes this code more
    compatible with the IO in SEPlib
    """
    hname = socket.gethostname()
    if(alias):
      with open('/etc/hosts','r') as f:
        for line in f.readlines():
          cols = line.lower().split()
          if(len(cols) > 1):
            if(hname == cols[1] and len(cols) > 2):
              hname = cols[2]

    return hname

  def id_generator(self,size=6, chars=string.ascii_uppercase + string.digits):
    """ Creates a random string with uppercase letters and integers """
    return ''.join(random.choice(chars) for _ in range(size))

  def yn2zoo(self,yn):
    """ Converts a 'y' or 'n' to an integer """
    if(yn == "n"):
      zoo = 0
    else:
      zoo = 1

    return zoo

  def read_list(self,arg,default,dtype='int'):
    """ Reads in comma delimited string at the command line into a python list """
    if(len(arg) == 0):
      return default
    else:
      if(dtype == 'int'):
        olist = [int(val) for val in arg.split(',')]
        return olist
      elif(dtype == 'float'):
        olist = [float(val) for val in arg.split(',')]
        return olist
      else:
        print("Type %s not recognized. Returning default")
        return default

  ## Vplot plotting
  #TODO: For now, I write the file to wherever you are.
  #      This may not be good in the future
  def pltgrey(self,daxes,dat,greyargs=None,shwvplt=True,figname=None,bg=False,savehfile=False):
    """ Plots a numpy array using SEP Grey """
    assert(daxes.ndims > 1), "Only use Grey for arrays of ndim=2 or larger"
    # Create random output filename
    rgname = ""
    if(self.argv is None):
      rgname = "python" + self.id_generator() + "Grey"
    else:
      rgname = self.argv[0].split("/")[-1] + self.id_generator() + "Grey"
    rghname = rgname + ".H"
    self.write_file(rghname,daxes,dat)
    # Get output figname
    figfile = None; ext = None
    if(figname != None):
      figfile, ext = opsys.path.splitext(figname)
    gpltcmd = 'Grey < %s'%(rghname); gsvecmd = 'Grey < %s'%(rghname)

    ## Plot and write figure
    if(shwvplt and figname != None):
      if(greyargs != None):
        gpltcmd += " " + greyargs
        gsvecmd += " " + greyargs

        gpltcmd += " | Tube -geometry 600x500"
        if(bg): gpltcmd += " &"
        gsvecmd += " out=%s > /dev/null "%(figfile+".v")
      # Plot and write the vplot
      print(gsvecmd)
      sp.check_call(gpltcmd,shell=True)
      sp.check_call(gsvecmd,shell=True)
      if(savehfile == False):
        sp.check_call("Rm %s"%(rghname),shell=True)
      if(ext == ".v"):
        return
      msg = "Invalid image file extension %s. Only 'pdf','png','jpg' or 'tiff' allowed for now"%(ext)
      assert (ext == ".pdf" or ext == ".png" or ext == ".jpg" or ext == ".tiff"),msg
      # First convert to EPS and then PDF
      ps = "pstexpen %s %s color=Y fat=1 fatmult=1.5 invras=Y force="%(figfile+".v",figfile+".ps")
      pdf = "epstopdf %s"%(figfile+".ps")
      print(ps)
      sp.check_call(ps,shell=True)
      print(pdf)
      sp.check_call(pdf,shell=True)
      if(ext != ".pdf"):
        # Use Imagemagick to convert to other file type
        sp.check_call("convert %s %s"%(figfile+".pdf",figname),shell=True)

    ## Just plot the figure
    elif(shwvplt and figname == None):
      if(greyargs != None):
        gpltcmd += " " + greyargs

      gpltcmd += " | Tube -geometry 600x500"
      if(bg): gpltcmd += " &"

      print(gpltcmd)
      # Plot the figure with vplot
      sp.check_call(gpltcmd,shell=True)
      if(savehfile == False):
        sp.check_call("Rm %s"%(rghname),shell=True)

    ## Just write the figure
    elif(shwvplt == False and figname != None):
      if(greyargs != None):
        gsvecmd += " " + greyargs

        gsvecmd += " out=%s > /dev/null "%(figfile+".v")
      # Write the vplot
      print(gsvecmd)
      sp.check_call(gsvecmd,shell=True)
      if(savehfile == False):
        sp.check_call("Rm %s"%(rghname),shell=True)
      if(ext == ".v"):
        return
      msg = "Invalid image file extension %s. Only 'pdf','png','jpg' or 'tiff' allowed for now"%(ext)
      assert (ext == ".pdf" or ext == ".png" or ext == ".jpg" or ext == ".tiff"),msg
      # First convert to EPS and then PDF
      ps = "pstexpen %s %s color=Y fat=1 fatmult=1.5 invras=Y force="%(figfile+".v",figfile+".ps")
      pdf = "epstopdf %s"%(figfile+".ps")
      sp.check_call(ps,shell=True)
      print(ps)
      sp.check_call(pdf,shell=True)
      print(pdf)
      if(ext != "pdf"):
        # Use Imagemagick to convert to other file type
        sp.check_call("convert %s %s"%(figfile+".pdf",figname),shell=True)

    return

  def pltgreyimg(self,dat,greyargs=None,o1=None,o2=None,d1=None,d2=None,bg=False,savehfile=False):
    """ Plot a Grey movie within Python """
    argdict = locals()
    assert(len(dat.shape) > 1), "Only use Grey for arrays of ndim=2 or larger"
    # Create random output filename
    rgname = ""
    if(self.argv is None):
      rgname = main.__file__ + self.id_generator() + "Grey"
    else:
      rgname = self.argv[0].split("/")[-1] + self.id_generator() + "Grey"
    rghname = rgname + ".H"
    # Build axes
    ns = list(dat.shape); os = []; ds = []
    for i in range(2):
      okey = 'o' + str(i+1)
      if(argdict[okey] != None):
        os.append(argdict[okey])
      else:
        os.append(0.0)
      dkey = 'd' + str(i+1)
      if(argdict[dkey] != None):
        ds.append(argdict[dkey])
      else:
        ds.append(1.0)
    daxes = axes(ns,os,ds)
    self.write_file(None,daxes,dat,ofname=rghname)
    # Build Grey command for viewing
    gpltcmd = 'Grey < %s'%(rghname); gsvecmd = 'Grey < %s'%(rghname)

    if(greyargs != None):
      gpltcmd += " " + greyargs

    gpltcmd += " | Tube -geometry 600x500"
    if(bg): gpltcmd += " &"

    print(gpltcmd)
    # Plot the figure with vplot
    sp.check_call(gpltcmd,shell=True)
    if(savehfile == False):
      sp.check_call("Rm %s"%(rghname),shell=True)

    return

  def pltgreymovie(self,dat,greyargs=None,o1=None,o2=None,o3=None,d1=None,d2=None,d3=None,bg=False,savehfile=False):
    """ Plot a Grey movie within Python """
    argdict = locals()
    assert(len(dat.shape) > 1), "Only use Grey for arrays of ndim=2 or larger"
    # Create random output filename
    rgname = ""
    if(self.argv is None):
      rgname = "python" + self.id_generator() + "Grey"
    else:
      rgname = self.argv[0].split("/")[-1] + self.id_generator() + "Grey"
    rghname = rgname + ".H"
    # Build axes
    ns = list(dat.shape); os = []; ds = []
    for i in range(3):
      okey = 'o' + str(i+1)
      if(argdict[okey] != None):
        os.append(argdict[okey])
      else:
        os.append(0.0)
      dkey = 'd' + str(i+1)
      if(argdict[dkey] != None):
        ds.append(argdict[dkey])
      else:
        ds.append(1.0)
    daxes = axes(ns,os,ds)
    self.write_file(None,daxes,dat,ofname=rghname)
    # Build Grey command for viewing
    gpltcmd = 'Grey < %s'%(rghname); gsvecmd = 'Grey < %s'%(rghname)

    if(greyargs != None):
      gpltcmd += " " + greyargs

    gpltcmd += " | Tube -geometry 600x500"
    if(bg): gpltcmd += " &"

    print(gpltcmd)
    # Plot the figure with vplot
    sp.check_call(gpltcmd,shell=True)
    if(savehfile == False):
      sp.check_call("Rm %s"%(rghname),shell=True)

    return

  def pltgraph(self,daxess,dats,graphargs=None,shwvplt=True,figname=None,bg=False,savehfile=False):
    """ Plots a numpy array using SEP Graph """
    numplots = len(daxess)
    n1 = daxess[0].n[0]; o1 = daxess[0].o[0]; d1 = daxess[0].d[0]
    origs = []; samps = []
    for iaxes in daxess:
      assert(iaxes.ndims < 2), "Only use Graph for arrays of ndim=1"
      assert(iaxes.n[0] == n1), "All axes must be same for multiple graphs"
    # Create new file for multiple plots
    oaxes = axes([n1,numplots],[o1,0.0],[d1,1.0])
    odat = np.zeros(oaxes.n)
    for ipl in range(numplots):
      odat[:,ipl] = dats[ipl]
    # Create random output filename
    rgname = ""
    if(self.argv is None):
      rgname = "python" + self.id_generator() + "Graph"
    else:
      rgname = self.argv[0].split("/")[-1] + self.id_generator() + "Graph"
    rghname = rgname + ".H"
    self.write_file(None,oaxes,odat,ofname=rghname)
    # Get output figname
    figfile = None; ext = None
    if(figname != None):
      figfile, ext = opsys.path.splitext(figname)
    gpltcmd = 'Graph < %s'%(rghname); gsvecmd = 'Graph < %s'%(rghname)

    ## Plot and write figure
    if(shwvplt and figname != None):
      if(graphargs != None):
        gpltcmd += " " + graphargs
        gsvecmd += " " + graphargs

        gpltcmd += " | Tube -geometry 600x500"
        if(bg): gpltcmd += " &"
        gsvecmd += " out=%s > /dev/null "%(figfile+".v")
      # Plot and write the vplot
      print(gsvecmd)
      sp.check_call(gpltcmd,shell=True)
      sp.check_call(gsvecmd,shell=True)
      if(savehfile == False):
        sp.check_call("Rm %s"%(rghname),shell=True)
      if(ext == ".v"):
        return
      msg = "Invalid image file extension %s. Only 'pdf','png','jpg' or 'tiff' allowed for now"%(ext)
      assert (ext == ".pdf" or ext == ".png" or ext == ".jpg" or ext == ".tiff"),msg
      # First convert to EPS and then PDF
      ps = "pstexpen %s %s color=Y fat=1 fatmult=1.5 invras=Y force="%(figfile+".v",figfile+".ps")
      pdf = "epstopdf %s"%(figfile+".ps")
      print(ps)
      sp.check_call(ps,shell=True)
      print(pdf)
      sp.check_call(pdf,shell=True)
      if(ext != ".pdf"):
        # Use Imagemagick to convert to other file type
        sp.check_call("convert %s %s"%(figfile+".pdf",figname),shell=True)

    ## Just plot the figure
    elif(shwvplt and figname == None):
      if(graphargs != None):
        gpltcmd += " " + graphargs

      gpltcmd += " | Tube -geometry 600x500"
      if(bg): gpltcmd += " &"

      print(gpltcmd)
      # Plot the figure with vplot
      sp.check_call(gpltcmd,shell=True)
      if(savehfile == False):
        sp.check_call("Rm %s"%(rghname),shell=True)

    ## Just write the figure
    elif(shwvplt == False and figname != None):
      if(graphargs != None):
        gsvecmd += " " + graphargs

        gsvecmd += " out=%s > /dev/null "%(figfile+".v")
      # Write the vplot
      print(gsvecmd)
      sp.check_call(gsvecmd,shell=True)
      if(savehfile == False):
        sp.check_call("Rm %s"%(rghname),shell=True)
      if(ext == ".v"):
        return
      msg = "Invalid image file extension %s. Only 'pdf','png','jpg' or 'tiff' allowed for now"%(ext)
      assert (ext == ".pdf" or ext == ".png" or ext == ".jpg" or ext == ".tiff"),msg
      # First convert to EPS and then PDF
      ps = "pstexpen %s %s color=Y fat=1 fatmult=1.5 invras=Y force="%(figfile+".v",figfile+".ps")
      pdf = "epstopdf %s"%(figfile+".ps")
      sp.check_call(ps,shell=True)
      print(ps)
      sp.check_call(pdf,shell=True)
      print(pdf)
      if(ext != "pdf"):
        # Use Imagemagick to convert to other file type
        sp.check_call("convert %s %s"%(figfile+".pdf",figname),shell=True)

    return

  def pltdots(self,daxes,dat,dotargs=None,shwvplt=True,figname=None,bg=False,savehfile=False):
    """ Plots a numpy array using SEP Dots"""
    assert(daxes.ndims < 2), "Only use Dots for arrays of ndim=1"
    # Create random output filename
    rgname = ""
    if(self.argv is None):
      rgname = main.__file__ + self.id_generator() + "Dots"
    else:
      rgname = self.argv[0].split("/")[-1] + self.id_generator() + "Dots"
    rghname = rgname + ".H"
    self.write_file(None,daxes,dat,ofname=rghname)
    # Get output figname
    figfile = None; ext = None
    if(figname != None):
      figfile, ext = opsys.path.splitext(figname)
    gpltcmd = 'Dots < %s'%(rghname); gsvecmd = 'Dots < %s'%(rghname)

    ## Plot and write figure
    if(shwvplt and figname != None):
      if(dotargs != None):
        gpltcmd += " " + dotargs
        gsvecmd += " " + dotargs

        gpltcmd += " | Tube -geometry 600x500"
        if(bg): gpltcmd += " &"
        gsvecmd += " out=%s > /dev/null "%(figfile+".v")
      # Plot and write the vplot
      print(gsvecmd)
      sp.check_call(gpltcmd,shell=True)
      sp.check_call(gsvecmd,shell=True)
      if(savehfile == False):
        sp.check_call("Rm %s"%(rghname),shell=True)
      if(ext == ".v"):
        return
      msg = "Invalid image file extension %s. Only 'pdf','png','jpg' or 'tiff' allowed for now"%(ext)
      assert (ext == ".pdf" or ext == ".png" or ext == ".jpg" or ext == ".tiff"),msg
      # First convert to EPS and then PDF
      ps = "pstexpen %s %s color=Y fat=1 fatmult=1.5 invras=Y force="%(figfile+".v",figfile+".ps")
      pdf = "epstopdf %s"%(figfile+".ps")
      print(ps)
      sp.check_call(ps,shell=True)
      print(pdf)
      sp.check_call(pdf,shell=True)
      if(ext != ".pdf"):
        # Use Imagemagick to convert to other file type
        sp.check_call("convert %s %s"%(figfile+".pdf",figname),shell=True)

    ## Just plot the figure
    elif(shwvplt and figname == None):
      if(dotargs != None):
        gpltcmd += " " + dotargs

      gpltcmd += " | Tube -geometry 600x500"
      if(bg): gpltcmd += " &"

      print(gpltcmd)
      # Plot the figure with vplot
      sp.check_call(gpltcmd,shell=True)
      if(savehfile == False):
        sp.check_call("Rm %s"%(rghname),shell=True)

    ## Just write the figure
    elif(shwvplt == False and figname != None):
      if(dotargs != None):
        gsvecmd += " " + dotargs

        gsvecmd += " out=%s > /dev/null "%(figfile+".v")
      # Write the vplot
      print(gsvecmd)
      sp.check_call(gsvecmd,shell=True)
      if(savehfile == False):
        sp.check_call("Rm %s"%(rghname),shell=True)
      if(ext == ".v"):
        return
      msg = "Invalid image file extension %s. Only 'pdf','png','jpg' or 'tiff' allowed for now"%(ext)
      assert (ext == ".pdf" or ext == ".png" or ext == ".jpg" or ext == ".tiff"),msg
      # First convert to EPS and then PDF
      ps = "pstexpen %s %s color=Y fat=1 fatmult=1.5 invras=Y force="%(figfile+".v",figfile+".ps")
      pdf = "epstopdf %s"%(figfile+".ps")
      sp.check_call(ps,shell=True)
      print(ps)
      sp.check_call(pdf,shell=True)
      print(pdf)
      if(ext != "pdf"):
        # Use Imagemagick to convert to other file type
        sp.check_call("convert %s %s"%(figfile+".pdf",figname),shell=True)

    return

