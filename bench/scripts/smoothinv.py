import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from opt.linopt.essops import identity, tcaijit
from opt.linopt.cd import cd
from utils.movie import viewpltframeskey

# IO
sep = seppy.sep()

# Build filter
nf = 2
flt = np.asarray([1,-1],dtype='float32')

idop = identity.identity()

nd = 50
dat = np.zeros(nd,dtype='float32')
dat[25] = 1.0

nr = nd + nf - 1 
tcaiop = tcaijit.tcai(nd,nr,flt)

rdat = np.zeros(nr,dtype='float32')

mod0 = np.zeros(nd,dtype='float32')

mods = []; objs = []
mod = cd(idop,dat,mod0,regop=tcaiop,rdat=rdat,eps=10.0,niter=40,mods=mods,objs=objs)

viewpltframeskey(np.asarray(mods),wbox=10,hbox=4)

