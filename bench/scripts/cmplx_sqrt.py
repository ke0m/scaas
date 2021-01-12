import numpy as np
import matplotlib.pyplot as plt

nx,ny = 500,1
dx,dy = 0.025,0.025

px,py = 112,0
bx,by = nx+px,ny+py

kk = np.zeros([by,bx],dtype='complex64')

# Spatial frequency axes
dkx = 2*np.pi/(bx*dx)
okx = 0 if bx == 1 else -np.pi/dx
dky = 2*np.pi/(by*dy)
oky = 0 if by == 1 else -np.pi/dy

kk = np.zeros([by,bx],dtype='float32')

# Populate the array
for iy in range(by):
  jy = iy+by//2 if iy < by//2 else iy-by//2
  ky = oky + jy*dky
  for ix in range(bx):
    jx = ix+bx//2 if ix < bx//2 else ix-bx//2
    kx = okx + jx*dkx
    kk[iy,ix] = kx*kx + ky*ky

sloref = 0.442970

w = np.complex(0.0,2*np.pi)
w2 = w*w

print(w2)
arg = w2*sloref + kk
cc = np.sqrt(w2*sloref + kk)

#plt.plot(cc[0]); plt.show()

out = np.zeros([by,bx],dtype='complex64')

sign = np.sign(np.imag(w))
idx = arg >= 0
out[idx]  = np.sqrt(arg[idx])
out[~idx] = sign*np.sqrt(-arg[~idx])*1j

#for iky in range(by):
#  for ikx in range(bx):
#    if(arg[iky,ikx] >= 0):
#      out[iky,ikx] = np.complex(np.sqrt(arg[iky,ikx]),0)
#    else:
#      out[iky,ikx] = np.complex(0,np.sign(np.imag(w))*np.sqrt(-arg[iky,ikx]))

plt.figure(); plt.plot(np.real(out[0])); plt.show()
plt.figure(); plt.plot(np.imag(out[0])); plt.show()

