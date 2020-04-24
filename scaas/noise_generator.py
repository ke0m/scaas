"""
A Perlin noise generator. Generates Perlin noise up to 3D

@author: Anu Chandran
@version: 2020.04.17
"""
import numpy as np
from numpy import random
from collections import Iterable
from multiprocessing import Pool, cpu_count

def fade(t):
    """ 
    An ease curve to smooth the transition between gradients
    6t^5 - 15t^4 + 10t^3
    """
    return t*t*t*(t*(6*t-15)+10)

def lerp(a,b,c):
    """ Linear interpolation """
    return a+c*(b-a)

def dotp(a,b):
    arr=zip(a,b)
    return( np.array([np.dot(*v) for v in arr]))

def hasher(index):
    return np.array([hash(i) for i in zip(*index)])

# outside just to permute the order on definition (not on call)
grads=[(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),(1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),(0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]
grads=np.random.permutation(grads)
def grad_3d(i):
    return grads[i]

def generate_grad_map_linear(N):
    return (np.array([grad_3d(random.randint(len(grads))) for i in range(N)]))

def perlin_3d_hash(x=np.array([0.5]),y=np.array([0.5]),z=np.array([0.5]),period=2,Ngrad=80,amp=128):
    """
    Creates perlin noise for a single octave
    """
    grads = generate_grad_map_linear(Ngrad)
    # Prepare for tiling
    xfull=x.copy(); yfull=y.copy(); zfull=z.copy()
    x=x[x<=period]; y=y[y<=period]; z=z[z<=period]
    tile_x=int(np.ceil(len(xfull)*1.0/len(x)))
    tile_y=int(np.ceil(len(yfull)*1.0/len(y)))
    tile_z=int(np.ceil(len(zfull)*1.0/len(z)))

    # Build the grid
    xg,yg,zg=np.meshgrid(x,y,z)
    xg=xg.flatten(); yg=yg.flatten(); zg=zg.flatten()
    ix=xg.astype('int'); iy=yg.astype('int'); iz=zg.astype('int')
    xf=xg-ix
    yf=yg-iy
    zf=zg-iz
    smxf=fade(xf)
    smyf=fade(yf)
    smzf=fade(zf)

    # Get gradients
    aaa=grads[ hasher( ((iz  )%period,(iy  )%period,(ix  )%period) )%Ngrad ]
    aab=grads[ hasher( ((iz  )%period,(iy  )%period,(ix+1)%period) )%Ngrad ]
    aba=grads[ hasher( ((iz  )%period,(iy+1)%period,(ix  )%period) )%Ngrad ]
    abb=grads[ hasher( ((iz  )%period,(iy+1)%period,(ix+1)%period) )%Ngrad ]
    baa=grads[ hasher( ((iz+1)%period,(iy  )%period,(ix  )%period) )%Ngrad ]
    bab=grads[ hasher( ((iz+1)%period,(iy  )%period,(ix+1)%period) )%Ngrad ]
    bba=grads[ hasher( ((iz+1)%period,(iy+1)%period,(ix  )%period) )%Ngrad ]
    bbb=grads[ hasher( ((iz+1)%period,(iy+1)%period,(ix+1)%period) )%Ngrad ]

    x1=lerp(dotp(aaa,np.array([zf  ,yf  ,xf]).T),dotp(aab,np.array([zf  ,yf  ,xf-1]).T),smxf)
    x2=lerp(dotp(aba,np.array([zf  ,yf-1,xf]).T),dotp(abb,np.array([zf  ,yf-1,xf-1]).T),smxf)
    y1=lerp(x1,x2,smyf)

    x3=lerp(dotp(baa,np.array([zf-1,yf  ,xf  ]).T),dotp(bab,np.array([zf-1,yf  ,xf-1]).T),smxf)
    x4=lerp(dotp(bba,np.array([zf-1,yf-1,xf  ]).T),dotp(bbb,np.array([zf-1,yf-1,xf-1]).T),smxf)
    y2=lerp(x3,x4,smyf)

    ave=lerp(y1,y2,smzf)
    ave=(amp*(ave+1)/2.0).reshape(len(z),len(y),len(x))

    # Tile the noise
    tiled_noise=np.tile(ave,(tile_z,tile_y,tile_x))
    ave=tiled_noise[0:len(zfull),0:len(yfull),0:len(xfull)]

    return ave

def perlin_wrapper(kwargs):
    """ Wrapper for parallelization using multiprocessing """
    random.seed()
    return perlin_3d_hash(**kwargs)

def perlin(x=np.array([0.5]),y=np.array([0.5]),z=np.array([0.5]),period=10.0,amp=1.0,persist=0.5,octaves=7, Ngrad=80, ncpu=cpu_count()):
    """ 
    Perlin noise generator 

      x       - the gradient grid for the fast axis 
      y       - the gradient grid for the middle axis
      z       - the gradient grid for the slow axis
      period  - determines the periodicity of the noise [10.0]
      amp     - a scaling factor to increase the amplitude of the noise [1.0]
      persist - the influence of each success octave [0.5]
      octaves - one set of noise. An increase in one octave doubles the frequency content [7]
      Ngrad   - Unsure of what this does really...
      ncpu    - Number of CPUs to use for parallelization
    """
    x=np.array(x); y=np.array(y); z=np.array(z)
    maxval=amp*np.sum([persist**octave for octave in range(octaves)])
    # Set up args for multiprocessing
    args=[{'x':x*(2**octave),'y':y*(2**octave),'z':z*(2**octave),'amp':amp*(persist)**octave,'period':period*2**octave, 'Ngrad':Ngrad} 
        for octave 
        in range(octaves)]
    if(ncpu > 1):
        run_pool=Pool(min(octaves,ncpu))
        out=run_pool.map(perlin_wrapper,args)
        run_pool.close()
        noise=(np.sum(np.array(out),axis=0)/maxval).squeeze()
    else:
      out = []
      # Loop and sum over octaves
      for octave in range(octaves):
        out.append(perlin_3d_hash(x=x*(2**octave), y=y*(2**octave), z=z*(2**octave), 
                   amp=amp*(persist)**octave, period=period*2**octave, Ngrad=Ngrad))
      noise=(np.sum(np.array(out),axis=0)/maxval).squeeze()
    return noise

