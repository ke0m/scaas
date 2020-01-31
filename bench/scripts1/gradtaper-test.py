import numpy as np
import inpout.seppy as seppy
from scaas.gradtaper import build_taper
import matplotlib.pyplot as plt

sep = seppy.sep([])

gaxes,grad = sep.read_file(None,ifname='marmgrad.H')
grad = grad.reshape(gaxes.n,order='F')
nz = gaxes.n[0]; nx = gaxes.n[1]

gmin = np.min(grad); gmax = np.max(grad)
plt.figure()
plt.imshow(grad,cmap='jet',vmin=0.7*gmin,vmax=0.7*gmax)

gtap1,gtap2 = build_taper(nx,nz,55,70)
tapped = gtap2*grad

plt.figure()
plt.plot(gtap1)
plt.plot(grad[:,100]/np.max(grad[:,100]))
plt.plot(tapped[:,100]/np.max(tapped[:,100]))

plt.figure()
plt.imshow(gtap2,cmap='jet')

sep.write_file(None,gaxes,tapped,ofname='tapped.H')

plt.figure()
nxu = 461; nzu = 151
plt.imshow(tapped[50+5:50+5+nzu,50+5:50+5+nxu],cmap='jet')

plt.show()
