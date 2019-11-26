import numpy as np

def build_taper(nx,nz,z1,z2):
  tap1d = np.zeros(nz,dtype='float32')
  if(z1 != 0 and z2 != 0):
    # Create taper in depth
    for iz in range(nz):
      if(iz < z1):
        tap1d[iz] = 0.0
      elif(iz >= z1 and iz <= z2):
        tap1d[iz] = np.cos(np.pi/2*(z2-iz)/(z2-z1))**2
      elif(iz >= z2):
        tap1d[iz] = 1.0
  else:
    tap1d[:] = 1.0
  return tap1d,np.tile(np.array([tap1d]).T,(1,nx))

