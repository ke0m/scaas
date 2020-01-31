import numpy as np

def matvec(A,x,b,g,r,dr):
  r[:]  = np.dot(A,x) - b
  g[:]  = np.dot(A.T,r)
  dr[:] = np.dot(A,g)

  return 0.5*np.dot(r,r)


