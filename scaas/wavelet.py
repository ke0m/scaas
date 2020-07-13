"""
Source time functions for wave propagation
@author: Joseph Jennings
@version: 2020.03.24
"""
import numpy as np

def ricker(nt,dt,f,amp,dly):
  """ 
  Given a time axis, dominant frequency and amplitude,
  returns a ricker wavelet

  Parameters
    nt  - length of output wavelet
    dt  - sampling rate of wavelet
    f   - dominant frequency of ricker wavelet
    amp - maximum amplitude of wavelet
    dly - time delay of wavelet
  """
  src = np.zeros(nt,dtype='float32')
  for it in range(nt):
    t = it*dt - dly
    pift = (np.pi*f*t)**2
    src[it] = amp*(1-2*pift)*np.exp(-pift)

  return src

def bandpass(nt,dt,fs,amp,dly):
  """
  Generates a bandpass wavelet in the frequency domain
  Basically a copy of Ali Almomin's Wavelet.f90

  Parameters
    nt  - length of output wavelet
    dt  - sampling rate of wavelet
    fs  - list of four frequencies that define band
    amp - maximum amplitude of wavelet
    dly - time delay of wavelet
  """
  df = 1/(nt*dt)
  shift = dly/dt
  wavw = np.zeros(nt,dtype='float32')

  # Build spectrum in frequency domain
  for i in range(int(nt/2)):
    f = (i-1)*df
    if(f < fs[0]):
      wavw[i] = 0
    elif(f >= fs[0] and f < fs[1]):
      wavw[i] = np.cos(np.pi/2*(fs[1]-f)/(fs[1]-fs[0]))**2
    elif(f >= fs[1] and f < fs[2]):
      wavw[i] = 1.0
    elif(f >= fs[2] and f < fs[3]):
      wavw[i] = np.cos(np.pi/2*(f-fs[2])/(fs[3]-fs[2]))**2
    else:
      wavw[i] = 0

  # Inverse FFT
  wavtsh = np.real(np.fft.ifft(wavw))
  wavt = amp*np.fft.fftshift(wavtsh)/np.max(wavtsh)

  # Apply delay
  t0 = int((nt/2))*dt
  wavtd = np.roll(wavt,int((t0+dly)/dt))

  return wavtd

