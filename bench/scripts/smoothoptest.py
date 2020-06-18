from scaas.trismooth import smoothop
import numpy as np
import matplotlib.pyplot as plt

#smop = smoothop([100,100],rect1=20,rect2=20)
smop = smoothop([100],rect1=20)

#mod = np.random.rand(100,100).astype('float32')
#dat = np.zeros(mod.shape,dtype='float32')

#smop.forward(False,mod,dat)

imp = np.zeros([100],dtype='float32')
imp[49] = 1.0
impsm = np.zeros([100],dtype='float32')

smop.forward(False,imp,impsm)

plt.figure(); plt.plot(imp)
plt.figure(); plt.plot(impsm)
plt.show()

#smop.dottest(add=False)



