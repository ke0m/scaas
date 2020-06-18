import inpout.seppy as seppy
import numpy as np
from opt.linopt.essops.weight import weight
from opt.linopt.cd import cd
from scaas.trismooth import smoothop
import matplotlib.pyplot as plt

sep = seppy.sep()

n = 100
x = np.zeros([n,n],dtype='float32')
x0 = np.zeros([n,n],dtype='float32')
y = np.zeros([n,n],dtype='float32')
z = np.zeros([n,n],dtype='float32')

#x[:] = 4; y[:] = 2
x[:] = np.random.rand(n,n)
y[:] = np.random.rand(n,n)

sep.write_file('num.H',x)
sep.write_file('den.H',y)

wop = weight(y)

smop = smoothop([n,n],rect1=5,rect2=5)

div = cd(wop,x,x0,shpop=smop,niter=100)

sep.write_file('mydiv.H',div)

