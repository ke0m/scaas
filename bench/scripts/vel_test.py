import scaas.velocity as vel
import matplotlib.pyplot as plt

nz = 200; dz = 20
nx = 400; dx = 20

vels = [1500,2200,2800]
z0s  = [65,130]

myvel,myref = vel.create_layered(nz,nx,dz,dx,z0s,vels,flat=True)

plt.figure(1)
plt.imshow(myvel,cmap='jet')

plt.figure(2)
plt.imshow(myref,cmap='gray',vmin=-1,vmax=1)
plt.show()
