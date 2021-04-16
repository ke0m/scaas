import argparse
import numpy as np
from scaas.wavelet import ricker
import scaas.oway.defaultgeom as geom
from scaas.trismooth import smooth


def main(args):

  # Build input slowness
  vz = 1.5 + np.linspace(0.0, args.dz * (args.nz - 1), args.nz)
  vel = np.ascontiguousarray(np.repeat(vz[np.newaxis, :], args.nx,
                                       axis=0).T).astype('float32')

  # Build reflectivity
  ref = np.zeros(vel.shape, dtype='float32')
  ref[349, 49:449] = 1.0
  npts = 25
  refsm = smooth(smooth(smooth(ref, rect1=npts), rect1=npts), rect1=npts)

  # Create ricker wavelet
  wav = ricker(
      n1=args.nt,
      d1=args.dt,
      freq=args.freq,
      amp=args.amp,
      dly=args.dly,
  )

  # Wave equation object
  wei = geom.defaultgeom(
      nx=args.nx,
      dx=args.dx,
      ny=1,
      dy=args.dx,
      nz=args.nz,
      dz=args.dz,
      nsx=args.nsx,
      dsx=args.dsx,
      osx=args.osx,
      nsy=1,
      dsy=1.0,
  )

  # Modeling
  dat = wei.model_data(
      wav,
      args.dt,
      args.dly,
      minf=args.minf,
      maxf=args.maxf,
      vel=vel,
      ref=refsm,
      time=True,
      ntx=15,
      px=112,
      nrmax=3,
      nthrds=args.nthrds,
      sverb=False,
      wverb=True,
      eps=0.0,
  )

  # Imaging
  img = wei.image_data(
      dat,
      args.dt,
      minf=args.minf,
      maxf=args.maxf,
      vel=vel,
      nthrds=args.nthrds,
      sverb=False,
      wverb=True,
  )


def attach_args(parser=argparse.ArgumentParser()):
  # Spatial parameters
  parser.add_argument("--nx", type=int, default=500)
  parser.add_argument("--dx", type=float, default=0.015)
  parser.add_argument("--nz", type=int, default=400)
  parser.add_argument("--dz", type=float, default=0.005)
  # Acquisition parameters
  parser.add_argument("--nsx", type=int, default=1)
  parser.add_argument("--dsx", type=float, default=10.0)
  parser.add_argument("--osx", type=float, default=250.0)
  # Wavelet parameters
  parser.add_argument("--nt", type=int, default=2000)
  parser.add_argument("--dt", type=float, default=0.004)
  parser.add_argument("--freq", type=float, default=8.0)
  parser.add_argument("--minf", type=float, default=1.0)
  parser.add_argument("--maxf", type=float, default=31.0)
  parser.add_argument("--dly", type=float, default=0.2)
  parser.add_argument("--nthrds", type=int, default=40)
  return parser


if __name__ == "__main__":
  main(attach_args().parse_args())
