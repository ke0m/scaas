import numpy as np

def ricker(nt,dt,f,amp,dly):
  """ Given a time axis, dominant frequency and amplitude,
  returns a ricker wavelet
  """
  src = np.zeros(nt,dtype='float32')
  for it in range(nt):
    t = it*dt - dly
    pift = (np.pi*f*t)**2
    src[it] = amp*(1-2*pift)*np.exp(-pift)

  return src
