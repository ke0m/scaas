from scaas.trismooth import smoothop
import numpy as np
import matplotlib.pyplot as plt

smop = smoothop([100,100],rect1=20,rect2=20)

mod = np.random.rand(100,100).astype('float32')
dat = np.zeros(mod.shape,dtype='float32')

#smop.forward(False,mod,dat)

smop.dottest()



