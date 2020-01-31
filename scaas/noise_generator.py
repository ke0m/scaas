# A perlin noise generator
# @author: Anu Chandran

import numpy as np
from numpy import random
from collections import Iterable
import timeit
from multiprocessing import Pool, cpu_count
import pylab
debug=False
#random.seed(7)
def fade(t):
    return t*t*t*(t*(6*t-15)+10)

def lerp(a,b,c):
    return a+c*(b-a)

def dotp(a,b):
     arr=zip(a,b)
     return( np.array([np.dot(*v) for v in arr]))

#def hasher(index):
#    #if isinstance(index[0],Iterable):
#        return np.array([hash(i) for i in zip(*index)])
#    #else:
#    #    return hash((x,y,z))
#
def hasher(index):
        return np.array([hash(i) for i in zip(*index)])
#

def grad_1d(i):
    grads=[1,-1]
    return(grads[i])
def grad_2d(i):
    grads=[(1,1),(1,-1),(-1,1),(-1,-1)]
    return(grads[i])
grads=[(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),(1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),(0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]
# just to permute the order on definition (not on call)
grads=np.random.permutation(grads)
def grad_3d(i):
    #grads=[(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),(1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),(0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]
    return(grads[i])

def generate_grad_map_linear(N,dim=3):
        return (np.array([grad_3d(random.randint(12)) for i in range(N)]))

def perlin_3d_hash(x=np.array([0.5]),y=np.array([0.5]),z=np.array([0.5]),period=2,Ngrad=80,amp=128):
    tstart=timeit.default_timer()
    grads=generate_grad_map_linear(Ngrad,dim=3)
    #prepare for tiling
    xfull=x.copy()
    yfull=y.copy()
    zfull=z.copy()
    x=x[x<=period]
    y=y[y<=period]
    z=z[z<=period]
    tile_x=int(np.ceil(len(xfull)*1.0/len(x)))
    tile_y=int(np.ceil(len(yfull)*1.0/len(y)))
    tile_z=int(np.ceil(len(zfull)*1.0/len(z)))
    if debug:
        print(timeit.default_timer()-tstart, 'grad map')
        print('grid shape raw=',x.shape,y.shape,z.shape)
    xg,yg,zg=np.meshgrid(x,y,z)
    xg=xg.flatten()
    yg=yg.flatten()
    zg=zg.flatten()
    if debug:
        print('grid shape=',xg.shape,yg.shape,zg.shape)
    ix=xg.astype('int')
    iy=yg.astype('int')
    iz=zg.astype('int')
    xf=xg-ix
    yf=yg-iy
    zf=zg-iz
    smxf=fade(xf)
    smyf=fade(yf)
    smzf=fade(zf)
    if debug:
        print(timeit.default_timer()-tstart, 'set up grid')

    aaa=grads[hasher((iz%period,iy%period,ix%period))%Ngrad]
    aab=grads[hasher((iz%period,iy%period,(ix+1)%period))%Ngrad]
    aba=grads[hasher((iz%period,(iy+1)%period,ix%period))%Ngrad]
    abb=grads[hasher((iz%period,(iy+1)%period,(ix+1)%period))%Ngrad]
    baa=grads[hasher(((iz+1)%period,iy%period,ix%period))%Ngrad]
    bab=grads[hasher(((iz+1)%period,iy%period,(ix+1)%period))%Ngrad]
    bba=grads[hasher(((iz+1)%period,(iy+1)%period,ix%period))%Ngrad]
    bbb=grads[hasher(((iz+1)%period,(iy+1)%period,(ix+1)%period))%Ngrad]
    if debug:
        print ('xf=',xf)
        print ('yf=',yf)
        print ('zf=',zf)
        print('aaa=',aaa)
        print('aab=',aab)
        print('aba=',aba)
        print('abb=',abb)
        print('baa=',baa)
        print('bab=',bab)
        print('bba=',bba)
        print('bbb=',bbb)
        print(timeit.default_timer()-tstart, 'grads')
    x1=lerp(dotp(aaa,np.array([zf,yf,xf]).T),dotp(aab,np.array([zf,yf,xf-1]).T),smxf)
    x2=lerp(dotp(aba,np.array([zf,yf-1,xf]).T),dotp(abb,np.array([zf,yf-1,xf-1]).T),smxf)
    y1=lerp(x1,x2,smyf)
    if debug:
        print(timeit.default_timer()-tstart, 'first y avg')
    x3=lerp(dotp(baa,np.array([zf-1,yf,xf]).T),dotp(bab,np.array([zf-1,yf,xf-1]).T),smxf)
    x4=lerp(dotp(bba,np.array([zf-1,yf-1,xf]).T),dotp(bbb,np.array([zf-1,yf-1,xf-1]).T),smxf)
    y2=lerp(x3,x4,smyf)
    if debug:
        print(timeit.default_timer()-tstart, 'second y avg')
    ave=lerp(y1,y2,smzf)
    ave=(amp*(ave+1)/2.0).reshape(len(z),len(y),len(x))
    if debug:
        print('x1=',x1)
        print('x2=',x2)
        print('x3=',x3)
        print('x4=',x4)
        print('y1=',y1)
        print('y2=',y2)
        print('ave=',ave)
        print(timeit.default_timer()-tstart, 'z avg')

    # tile the noise
    tiled_noise=np.tile(ave,(tile_z,tile_y,tile_x))
    ave=tiled_noise[0:len(zfull),0:len(yfull),0:len(xfull)]
    return ave
def perlin_wrapper(kwargs):
    random.seed()
    return perlin_3d_hash(**kwargs)
def perlin(x=np.array([0.5]),y=np.array([0.5]),z=np.array([0.5]),period=10.0,amp=1.0,persist=0.5,octaves=7, Ngrad=80, ncpu=cpu_count()):
    x=np.array(x)
    y=np.array(y)
    z=np.array(z)
    maxval=amp*np.sum([persist**octave for octave in range(octaves)])
    args=[{'x':x*(2**octave),'y':y*(2**octave),'z':z*(2**octave),'amp':amp*(persist)**octave,'period':period*2**octave, 'Ngrad':Ngrad} for octave in range(octaves)]
    if(ncpu > 1):
        run_pool=Pool(min(octaves,ncpu))
        out=run_pool.map(perlin_wrapper,args)
        run_pool.close()
        noise=(np.sum(np.array(out),axis=0)/maxval).squeeze()
    else:
      out = []
      for octave in range(octaves):
        out.append(perlin_3d_hash(x=x*(2**octave), y=y*(2**octave), z=z*(2**octave), amp=amp*(persist)**octave, period=period*2**octave, Ngrad=Ngrad))
      noise=(np.sum(np.array(out),axis=0)/maxval).squeeze()
    return (noise)

if __name__=='__main__':
    debug=False
    x=np.linspace(0,1,40)
    #y=x
    #z=x
    y=np.array([0.1])
    z=np.array([0.1])
    tstart=timeit.default_timer()
    period=1.0
    noise1=perlin(x=x,y=y,z=z,octaves=1,period=period, Ngrad=80)
    noise2=perlin(x=x,y=y,z=z,octaves=1,period=period, Ngrad=80)
    telapse=timeit.default_timer()-tstart
    pylab.plot(x,noise1)
    pylab.plot(x,noise2)
    print ('elapsed time=',telapse)
    #print (noise)
#    pylab.imshow(noise[20,:,:])
#    pylab.title('z slice')
#    pylab.figure()
#    pylab.imshow(noise[:,20,:])
#    pylab.title('y slice')
#    pylab.figure()
#    pylab.imshow(noise[:,:,20])
#    pylab.title('x slice')

    pylab.show()

