from __future__ import print_function
import numpy as np
import os,socket,getpass
import datetime
import string, random
import subprocess as sp

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

  def __init__(self,argv):
    self.hdict = {}
    self.haxes = {}
    self.argv  = argv
    self.hostname = socket.gethostname()

  def get_fname(self, tag):
    """ Gets the file name with the associated tag """
    fname = None
    for iarg in self.argv:
      keyval = iarg.split('=')
      if(len(keyval) != 2): continue
      if(keyval[0] == tag):
        fname = keyval[1]
        break

    return fname

  def read_header(self,tag,ifname=None):
    """ Reads a SEP header file from tag and returns the axes """
    self.hdict = {}
    # Get the filename with the given tag
    assert(tag != None or ifname != None), "Need either a tag or inputfile for reading.\n"
    if(tag != None):
      hin  = self.get_fname(tag)
      assert (hin != None), "No file associated with tag '%s'"%(tag)
    else:
      hin = ifname
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

  def read_header_dict(self,tag,ifname=None):
    """ Reads a SEP header file and returns a dictionary """
    hdict = {}
    # Get the filename with the given tag
    assert(tag != None or ifname != None), "Need either a tag or inputfile for reading.\n"
    if(tag != None):
      hin  = self.get_fname(tag)
      assert (hin != None), "No file associated with tag '%s'"%(tag)
    else:
      hin = ifname
    hout = self.get_fname("out")
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

  def read_file(self,tag,ifname=None,form='xdr'):
    """ Reads a SEP file from tag and returns the data and the axes """
    faxes  = self.read_header(tag,ifname)
    dat = np.zeros(faxes.get_nelem())
    with open(self.hdict["in"],'rb') as f:
      if(form == 'xdr'):
        dat[:] = np.fromfile(f, dtype='>f')
      elif(form == 'native'):
        dat[:] = np.fromfile(f, dtype='<f')
      else:
        print("Failed to read in file. Format %s not recognized\n"%(form))
  
    return faxes, dat

  def from_header(self,tag,keys,ifname=None):
    """ Given a list of keys (strings), returns the values from the header """
    vals = []
    # Read the header dictionary
    thdict = self.read_header_dict(tag,ifname)
    # Loop over all keys
    for ikey in keys:
      if ikey in thdict:
        vals.append(thdict[ikey])

    return vals

  def write_header(self,tag,ofaxes,ofname=None,form='xdr'):
    """ Writes header information to SEP file and returns
    the path to the output """
    fout = None
    assert(tag != None or ofname != None), "Need a tag or output file name to write a header."
    if(tag != None):
      ofname = self.get_fname(tag)

    if(tag == "out"):
      assert(ofname != None), "No output file name found. Did you pass argv to seppy?"
      fout = open(ofname,"a")
    else:
      fout = open(ofname,"w+")
    # Write the first line
    fout.write('\n' + self.get_fline()+'\n')
    # Get the datapath
    if(len(ofname.split('/')) > 1):
      ofname = ofname.split('/')[-1]
    opath = self.get_datapath() + ofname + "@"
    fout.write('\t\tsets next: in="%s"\n'%(opath))
    # Print axes
    for k in range(ofaxes.ndims):
      if(ofaxes.label != None): 
        fout.write("\t\tn%d=%d o%d=%f d%d=%.12f label%d=%s\n"%
            (k+1,ofaxes.n[k],k+1,ofaxes.o[k],k+1,ofaxes.d[k],k+1,ofaxes.label[k]))
      else:
        fout.write("\t\tn%d=%d o%d=%f d%d=%.12f\n"%
            (k+1,ofaxes.n[k],k+1,ofaxes.o[k],k+1,ofaxes.d[k]))

    if(form == 'xdr'):
      fout.write('\t\tdata_format="xdr_float" esize=4\n')
    elif(form == 'native'):
      fout.write('\t\tdata_format="native_float" esize=4\n')
    else:
      print("Error: format %s not recognized"%(form))

    fout.close()

    return opath

  #TODO: really inefficient for large files (lots of memory and slow)
  # probably should wrap Huy's code
  def write_file(self,tag,ofaxes,data,ofname=None,form='xdr'):
    """ Writes data and axes to a SEP header and binary """
    opath = self.write_header(tag,ofaxes,ofname,form)
    with open(opath,'wb') as f:
      if(form == 'xdr'):
        data.flatten('F').astype('>f').tofile(f)
      elif(form == 'native'):
        data.flatten('F').astype('<f').tofile(f)
      else:
        print("Failed to write file. Format %s not recognized\n"%(form))

  def to_header(self,tag,info,ofname=None):
    """ Writes any auxiliary information to header """
    fout = None
    assert(tag != None or ofname != None), "Need a tag or output file name to write a header."
    if(tag != None):
      ofname = self.get_fname(tag)

    # Open file header
    if(tag == "out"):
      assert(ofname != None), "No output file name found. Did you pass argv to seppy?"
      fout = open(ofname,"a")
    else:
      fout = open(ofname,"w+")
    fout.write('\n' + info)
    fout.close()

  def write_dummyaxis(self,tag,dim,ofname=None):
    """ Writes a single axis to a SEP header """
    if(ofname == None):
      ofname = self.get_fname(tag)

    fout = open(ofname,"a")
    fout.write("\n\t\tn%d=1 o%d=0.0 d%d=1.0\n"%(dim,dim,dim))
    fout.close()
  
  def append_to_movie(self,tag,ofaxes,data,niter,ofname=None,form='xdr'):
    """ Appends to a file for an inversion movie"""
    if(ofname == None):
      ofname = self.get_fname(tag)

    # Write an extra line to the header
    fout = open(ofname,"a")
    odim = ofaxes.ndims + 1
    fout.write("\n\t\tn%d=%d o%d=0.0 d%d=1.0\n"%(odim,niter,odim,odim))
    fout.close()

    # Append the data to the binary
    opath = self.get_datapath() + ofname + "@"
    with open(opath,'ab') as f:
      if(form == 'xdr'):
        data.flatten('F').astype('>f').tofile(f)
      elif(form == 'native'):
        data.flatten('F').astype('<f').tofile(f)

  def get_fline(self):
    """ Returns the first line of the program header """
    if(len(self.argv) == 0):
      fline = "python"
    else:
      fline = self.argv[0]
    # Get user and hostname
    username = getpass.getuser()
    fline += ":\t" + username + "@" + self.hostname + "\t\t"
    # Get time and date
    time = datetime.datetime.today()
    fline += time.strftime("%a %b %d %H:%M:%S %Y")

    return fline

  #TODO: only gets the first line. Need to search over the other
  # lines for the matching hostname
  def get_datapath(self):
    """ Gets the set datpath for writing SEP binaries """
    dpath = ''
    # Check if passed as argument
    dpath = self.get_fname("datapath")
    if(dpath == None):
      # Look in home directory
      datstring = os.environ['HOME'] + "/.datapath"
      if(os.path.exists(datstring) == True):
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
    elif(dpath == None and "DATAPATH" in os.environ):
      dpath = os.environ['DATAPATH']
    #else:
    #  dpath = '/tmp/'

    return dpath
  
  def id_generator(self,size=6, chars=string.ascii_uppercase + string.digits):
    """ Creates a random string with uppercase letters and integers """
    return ''.join(random.choice(chars) for _ in range(size))

  #TODO: For now, I write the file to wherever you are.
  #      This may not be good in the future
  def pltgrey(self,daxes,dat,greyargs=None,shwvplt=True,figname=None,bg=False,savehfile=False):
    """ Plots a numpy array using SEP Grey """
    assert(daxes.ndims > 1), "Only use Grey for arrays of ndim=2 or larger"
    # Create random output filename
    rgname = ""
    if(len(self.argv) == 0):
      rgname = "python" + self.id_generator() + "Grey"
    else:
      rgname = self.argv[0].split("/")[-1] + self.id_generator() + "Grey"
    rghname = rgname + ".H"
    self.write_file(None,daxes,dat,ofname=rghname)
    # Get output figname
    figfile = None; ext = None
    if(figname != None):
      figfile, ext = os.path.splitext(figname)
    gpltcmd = 'Grey < %s'%(rghname); gsvecmd = 'Grey < %s'%(rghname)

    ## Plot and write figure
    if(shwvplt and figname != None):
      if(greyargs != None):
        gpltcmd += " " + greyargs
        gsvecmd += " " + greyargs
    
        gpltcmd += " | Tube -geometry 600x500"
        if(bg): gpltcmd += "&"
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
      if(bg): gpltcmd += "&"

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
    if(len(self.argv) == 0):
      rgname = "python" + self.id_generator() + "Graph"
    else:
      rgname = self.argv[0].split("/")[-1] + self.id_generator() + "Graph"
    rghname = rgname + ".H"
    self.write_file(None,oaxes,odat,ofname=rghname)
    # Get output figname
    figfile = None; ext = None
    if(figname != None):
      figfile, ext = os.path.splitext(figname)
    gpltcmd = 'Graph < %s'%(rghname); gsvecmd = 'Graph < %s'%(rghname)

    ## Plot and write figure
    if(shwvplt and figname != None):
      if(graphargs != None):
        gpltcmd += " " + graphargs
        gsvecmd += " " + graphargs
    
        gpltcmd += " | Tube -geometry 600x500"
        if(bg): gpltcmd += "&"
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
      if(bg): gpltcmd += "&"

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
    if(len(self.argv) == 0):
      rgname = "python" + self.id_generator() + "Dots"
    else:
      rgname = self.argv[0].split("/")[-1] + self.id_generator() + "Dots"
    rghname = rgname + ".H"
    self.write_file(None,daxes,dat,ofname=rghname)
    # Get output figname
    figfile = None; ext = None
    if(figname != None):
      figfile, ext = os.path.splitext(figname)
    gpltcmd = 'Dots < %s'%(rghname); gsvecmd = 'Dots < %s'%(rghname)

    ## Plot and write figure
    if(shwvplt and figname != None):
      if(dotargs != None):
        gpltcmd += " " + dotargs 
        gsvecmd += " " + dotargs
    
        gpltcmd += " | Tube -geometry 600x500"
        if(bg): gpltcmd += "&"
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
      if(bg): gpltcmd += "&"

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

