import argparse
import numpy as np
from scaas.wavelet import ricker
import scaas.oway.defaultgeom as geom
from scaas.filter.trismooth import smooth
from plot import plot_3d, plot_imgpoff
import time


def main(args):

  # Build input slowness
  vz = 1.5 + np.linspace(0.0, args.dz * (args.nz - 1), args.nz)
  velx = np.ascontiguousarray(np.repeat(vz[np.newaxis, :], args.nx, axis=0))
  vel = np.ascontiguousarray(np.repeat(velx[np.newaxis, :, :], args.ny,
                                       axis=0)).T

  velin = np.zeros([args.nz, args.ny, args.nx], dtype='float32')
  velin[:] = vel[:]

  # Build reflectivity
  ref = np.zeros(velin.shape, dtype='float32')
  ref[349, 49:449, 49:449] = 1.0
  npts = 25
  refsm = smooth(smooth(smooth(ref, rect1=npts, rect2=npts),
                        rect1=npts,
                        rect2=npts),
                 rect1=npts,
                 rect2=npts)

  # Create ricker wavelet
  wav = ricker(args.nt, args.dt, args.freq, args.amp, args.dly)

  wei = geom.defaultgeom(
      nx=args.nx,
      dx=args.dx,
      ny=args.ny,
      dy=args.dy,
      nz=args.nz,
      dz=args.dz,
      nsx=1,
      dsx=50,
      osx=args.osx,
      nsy=1,
      dsy=50,
      osy=args.osy,
  )

  beg = time.time()
  dat = wei.model_data(
      wav,
      args.dt,
      args.dly,
      minf=args.minf,
      maxf=args.maxf,
      vel=velin,
      ref=refsm,
      ntx=15,
      px=112,
      nty=15,
      py=112,
      nthrds=args.nthrds,
      sverb=False,
      wverb=True,
  )

  img = wei.image_data(
      dat,
      args.dt,
      minf=args.minf,
      maxf=args.maxf,
      vel=velin,
      nthrds=args.nthrds,
      nhx=args.nhx,
      big=True,
      sverb=False,
      wverb=True,
  )
  print("Time Elapsed=%f s" % (time.time() - beg))

  plot_3d(
      img[0, 20],
      ds=[args.dz, args.dy, args.dx],
      loc3=args.img_zslice * args.dz,
      loc2=args.img_xslice * args.dx,
      loc1=args.img_yslice * args.dy,
      label1='X (km)',
      label2='Y (km)',
      label3='Z (km)',
      pclip=0.5,
      figname=args.output_img_fig,
  )
  # Extended image and y slice
  plot_imgpoff(
      np.expand_dims(img[:, :, :, args.img_yslice, :], axis=3),
      args.dx,
      args.dz,
      args.nhx,
      args.img_yslice,
      -args.nhx * args.dx,
      args.dx,
      figname=args.output_eimg_yslice_fig % (args.img_yslice),
  )
  # Extended image and x slice
  plot_imgpoff(
      np.expand_dims(img[:, :, :, :, args.img_xslice], axis=3),
      args.dx,
      args.dz,
      args.nhx,
      args.img_xslice,
      -args.nhx * args.dx,
      args.dx,
      figname=args.output_eimg_xslice_fig % (args.img_xslice),
  )


def attach_args(parser=argparse.ArgumentParser()):
  # Spatial parameters
  parser.add_argument("--nx", type=int, default=500)
  parser.add_argument("--dx", type=float, default=0.015)
  parser.add_argument("--ny", type=int, default=500)
  parser.add_argument("--dy", type=float, default=0.015)
  parser.add_argument("--nz", type=int, default=400)
  parser.add_argument("--dz", type=float, default=0.005)
  # Acqusition parameters
  parser.add_argument("--osx", type=float, default=250)
  parser.add_argument("--osy", type=float, default=250)
  # Wavelet parameters
  parser.add_argument("--nt", type=int, default=2000)
  parser.add_argument("--dt", type=float, default=0.004)
  parser.add_argument("--amp", type=float, default=0.5)
  parser.add_argument("--freq", type=float, default=8.0)
  parser.add_argument("--minf", type=float, default=1.0)
  parser.add_argument("--maxf", type=float, default=31.0)
  parser.add_argument("--dly", type=float, default=0.2)
  parser.add_argument("--nthrds", type=int, default=40)
  parser.add_argument("--nhx", type=int, default=20)
  # Output data figures
  parser.add_argument("--dat-xslice", type=int, default=250)
  parser.add_argument("--dat-yslice", type=int, default=250)
  parser.add_argument("--dat-tslice", type=int, default=450)
  # Output img figure
  parser.add_argument("--img-xslice", type=int, default=250)
  parser.add_argument("--img-yslice", type=int, default=250)
  parser.add_argument("--img-zslice", type=int, default=349)
  parser.add_argument("--output-img-fig", type=str, default="./eimg_3d.png")
  parser.add_argument(
      "--output-eimg-xslice-fig",
      type=str,
      default="./eimg_3d_xslc_%d.png",
  )
  parser.add_argument(
      "--output-eimg-yslice-fig",
      type=str,
      default="./eimg_3d_yslc_%d.png",
  )
  parser.add_argument("--show", action='store_true', default=False)
  parser.add_argument("--fsize", type=int, default=15)
  return parser


if __name__ == "__main__":
  main(attach_args().parse_args())
