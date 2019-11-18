import numpy as np
import matplotlib.pyplot as plt
def sinc_interp(x, s, u):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """

    assert(len(x) == len(s)), 'x and s must be same length'
    #if len(x) != len(s):
    #    raise Exception, 'x and s must be the same length'

    # Find the period
    T = s[1] - s[0]

    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    y = np.dot(x, np.sinc(sincM/T))
    return y



nt = 1000
ot = 0.0
dt = 0.01

t1 = np.linspace(ot,dt*(nt-1),nt)
t2 = np.linspace(ot,(0.5*dt)*(nt-1),nt)

f = np.sin(t1)
g = np.sin(t2)
plt.plot(f)
plt.plot(g)

plt.plot(sinc_interp(np.sin(t1),t1,t2))
plt.plot(np.interp(t2,t1,np.sin(t1)))
plt.show()
