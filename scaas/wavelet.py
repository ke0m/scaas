import numpy as np
from utils.signal import butter_bandpass_filter

def ricker(nt,dt,f,amp,dly):
  """ 
  Given a time axis, dominant frequency and amplitude,
  returns a ricker wavelet

  Parameters
    nt  - length of wavelet
    dt  - sampling interval
    f   - center frequency of wavelet
    amp - peak amplitude of wavelet
    dly - time delay of wavelet
  """
  src = np.zeros(nt,dtype='float32')
  for it in range(nt):
    t = it*dt - dly
    pift = (np.pi*f*t)**2
    src[it] = amp*(1-2*pift)*np.exp(-pift)

  return src

def butterworth(nt,dt,locut,hicut,amp,dly):
  """ 
  Creates a zero-phase butterworth wavelet 

  Parameters
    nt    - length of wavelet
    dt    - sampling interval
    hicut - high-cut of butterworth filter
    locut - low-cut of butterworth filter
    amp   - peak amplitude of wavelet
    dly   - time delay of wavelet
  """
  pass

def zerophase(nt,dt,f1,f2,f3,f4,amp,dly):
  """ 
  Creates a zero-phase wavelet based on tapered window function 

  Parameters
    nt  - length of wavelet
    dt  - sampling interval
    f1  - lowest frequency to be used in window
    f2  - low corner frequency for defining the taper
    f3  - high corner frequency for defining taper
    f4  - higest frequency in window
    amp - peak amplitude of wavelet
    dly - time delay of the wavelet
  """
  pass

