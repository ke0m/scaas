import numpy as np

def quad(x,g):
  """ 
  Simple quadratic function:
  minimum is -31 at x0=5 and x1=6
  """
  g[0] = 2*x[0] - x[1] - 4 
  g[1] = 2*x[1] - x[0] - 7 

  return x[0]**2 + x[1]**2 - x[0]*x[1] - 4*x[0] - 7*x[1]

def camel6(x,g):
  """
  Six hump camel function
  """
  x02 = x[0]*x[0]
  x04 = x02*x02
  x12 = x[1]*x[1]
  f = (4 - 2.1*x02 + x04/3.0) * x02 + x[0]*x[1] + 4*(x12 - 1)*x12
  g[0] = 8*x[0] - 8.2*x[0]*x02 + 2*x[0]*x04 + x[1]
  g[1] = x[0] + 16*x[1]*x12 - 8*x[1]

  return f

def rosenbrock(x,g):
  """
  Rosenbrock function
  """
  n = x.shape[0]
  f = 0
  for i in range(0,n,2):
    t1 = 1 - x[i]
    t2 = 1e1 * (x[i+1] - x[i]*x[i]);
    g[i+1] =  2e1*t2;
    g[i+0] = -2*(x[i] * g[i+1] + t1)
    f += t1*t1 + t2*t2;

  return f

def powell(x,g):
  n = x.shape[0]
  f = 0
  for i in range(0,n,4):
    # Function evaluation
    t1 = x[i+0] + 10*x[i+1]
    t2 = x[i+2] -    x[i+3]
    t3 = x[i+1] -  2*x[i+2]
    t4 = x[i+0] -    x[i+3]
    t33 = t3**3; t43 = t4**3
    f += t1**2 + 5*t2**2 + t33*t3 + 10*t43*t4
    # Gradient
    g[i+0] = 2*t1  + 40*t43
    g[i+1] = 20*t1 + 4*t33
    g[i+2] = 10*t2 - 8*t33
    g[i+3] = -10*t2 - 40*t43

  return f

def trid(x,g):
  n = x.shape[0]
  f = 0
  for i in range(n):
    t = x[i] - 1
    f += t*t
    g[i] = 2*t
  for i in range(n-1):
    f -= x[i]*x[i+1]
    g[i] -= x[i+1]
    g[i+1] -= x[i]

  return f

def rosenbrock1(x,g):
  n = x.shape[0]
  # Initialze f and g
  f = 0.0; g[:] = 0.0 
  for i in range(n-1):
    t1 = x[i] - 1
    t2 = x[i]*x[i] - x[i+1]
    f += t1*t1 + 100.0*t2*t2
    g[i+0] += 2.0*t1 + 400.0*t2*x[i]
    g[i+1] -= 200.0*t2

  return f


