import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from opt.linopt.essops import tcaijit,lintjit
from scaas.trismooth import smoothop
from opt.linopt.cd import cd
from utils.movie import viewpltframeskey

# IO
sep = seppy.sep()

# Read in coordinates and values
vaxes,vals = sep.read_file("lab3_valuew.H")
caxes,crds = sep.read_file("lab3_coordw.H")
vals = vals.astype('float32'); crds = crds.astype('float32')
nd = vals.shape[0]

# Build filter
nf = 2
flt = np.asarray([1,-1],dtype='float32')

#nf = 3
#flt = np.asarray([1,-2,1],dtype='float32')

#plt.plot(crds,vals); plt.show()

# Output regularized Model vector
nm = 200; om = 0.0; dm = 0.4
mod0 = np.zeros(nm,dtype='float32')

lintop = lintjit.lint(nm,om,dm,crds)

nr = nm + nf - 1
tcaiop = tcaijit.tcai(nm,nr,flt)

rdat = np.zeros(nr,dtype='float32')

smop = smoothop([nm],rect1=2)

mods = []; objs = []
mod1 = cd(lintop,vals,mod0,regop=tcaiop,rdat=rdat,eps=0.1,niter=100,mods=mods,objs=objs)
viewpltframeskey(np.asarray(mods),wbox=10,hbox=4)

mods2 = []; objs2 = []
mod2 = cd(lintop,vals,mod0,shpop=smop,niter=100,mods=mods2,objs=objs2)
viewpltframeskey(np.asarray(mods2),wbox=10,hbox=4)

