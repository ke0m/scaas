import numpy as np
from opt.linopt.essops.matmul import matmul
from opt.linopt.cd import cd
from scaas.trismooth import smoothop

n = 5
# Create lower triangular matrix
A = np.zeros([n,n],dtype='float32')

for irow in range(n):
  A[irow,:irow+1] = np.random.rand(irow+1)

print(A)

mop = matmul(A)

#mop.dottest(add=True)

b = np.random.rand(n)

xex = np.linalg.solve(A,b)

print(xex)

x0 = np.zeros(n,dtype='float32')
xsm = np.zeros(n,dtype='float32')

xme = cd(mop,b,x0,niter=10)

print(xme)

smop= smoothop([n],rect1=1)
#smop.forward(False,xme,xsm)
#print(xsm)
xshp = cd(mop,b,x0,shpop=smop,niter=10)

print(xshp)
