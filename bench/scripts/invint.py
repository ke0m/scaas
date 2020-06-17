import inpout.seppy as seppy
import numpy as np
import matplotlib.pyplot as plt
from opt.linopt.essops import tcaijit,lintjit
from opt.linopt.cd import cd

# IO
sep = seppy.sep()

# Read in coordinates and values
vaxes,vals = sep.read_file("lab3_valuew.H")
caxes,crds = sep.read_file("lab3_coordw.H")
nd = vals.shape[0]

# Build filter
nf = 2
flt = np.asarray([1,-1],dtype='float32')

#plt.plot(crds,vals); plt.show()

# Output regularized Model vector
nm = 200; om = 0.0; dm = 0.4

lintop = lintjit.lint(nm,om,dm,crds.astype('float32'))

nr = nm + nf - 1
tcaiop = tcaijit.tcai(nm,nr,flt)



