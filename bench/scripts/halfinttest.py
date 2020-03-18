import numpy as np
import scaas.halfint as halfint
import inpout.seppy as seppy
from resfoc.cosft import next_fast_size
import matplotlib.pyplot as plt

n = 101
spike = np.zeros([n],dtype='float32')
flt = np.zeros(n,dtype='float32')

nn = int(2*next_fast_size((n+1)/2))

spike[int(n/2)-1] = 1.0

pspike = np.zeros(nn,dtype='float32')

pspike[:n] = spike[:]

## Forward operator
#hiop = halfint.halfint(False,nn,1-1.0/n)
#hiop.forward(False,n,spike,flt)
#hiop.adjoint(False,n,flt,spike)

## Inverse operator
hiop = halfint.halfint(True,nn,1-1.0/n)
#hiop.forward(False,n,spike,flt)
hiop.adjoint(False,n,flt,spike)


# Read in Sergeys example
sep = seppy.sep([])
#daxes,dat = sep.read_file(None,ifname='spikehi.H')
#daxes,dat = sep.read_file(None,ifname='spikehiadj.H')

#daxes,dat = sep.read_file(None,ifname='spikeihi.H')
daxes,dat = sep.read_file(None,ifname='spikeihiadj.H')

plt.figure(1)
plt.plot(flt) 

plt.figure(2)
plt.plot(dat)

plt.show()


