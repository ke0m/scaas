"""
Taper functions. Useful for tapering the FWI gradient
@author: Joseph Jennings
@version: 2020.08.11
"""
import numpy as np
import matplotlib.pyplot as plt

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

def build_taper_bot(nx,nz,z1,z2):
  tap1d = np.zeros(nz,dtype='float32')
  if(z1 != 0 and z2 != 0):
    # Create taper in depth
    for iz in range(nz):
      if(iz < z1):
        tap1d[iz] = 1.0
      elif(iz >= z1 and iz <= z2):
        tap1d[iz] = np.sin(np.pi/2*(z2-iz)/(z2-z1))**2
      elif(iz >= z2):
        tap1d[iz] = 0.0
  else:
    tap1d[:] = 1.0
  return tap1d,np.tile(np.array([tap1d]).T,(1,nx))

def build_taper_ds(nx,nz,zt1,zb1,zt2,zb2):
  topt1,topt2 = build_taper(nx,nz,zt1,zb1)
  bott1,bott2 = build_taper_bot(nx,nz,zt2,zb2)
  return topt1*bott1,topt2*bott2

