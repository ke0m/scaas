"""
Adds Gaussian anomalies to a velocity model

@author: Joseph Jennings
@version: 2019.11.16
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import scipy.ndimage as flt

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    "scales": [],
    "rects": [],
    }
if args.conf_file:
  config = ConfigParser.SafeConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("Defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
parser.add_argument("in=",help="input velocity model")
parser.add_argument("out=",help="output velocity model with anomaly")
# Required arguments
requiredNamed = parser.add_argument_group('required parameters')
requiredNamed.add_argument("-pzs",help="Z positions of center of gaussian [iz1,iz2,...]",type=str)
requiredNamed.add_argument("-pxs",help="X positions of center of gaussian [ix1,ix2,...]",type=str)
# Optional arguments
parser.add_argument("-scales",help="Sizes of anomalies [sc1,sc2,...] (default is 100 m/s)",type=str)
parser.add_argument("-rects",help="Radii of anomalies [r1,r2,...] (default is 10)",type=str)
parser.add_argument("-verb",help="Verbosity flag (y or [n])")
args = parser.parse_args(remaining_argv)

# Set up SEP for IO
sep = seppy.sep(sys.argv)

# Get command line arguments
pxs = sep.read_list(args.pxs,[],dtype='int')
pzs = sep.read_list(args.pzs,[],dtype='int')
na = len(pxs)
assert(na == len(pzs)),"Length of pxs must equal length of pzs"

rects  = sep.read_list(args.rects,  np.zeros(na) + 6 , dtype='int')
scales = sep.read_list(args.scales, np.zeros(na) + 100, dtype='float')

# Verbosity
verb  = sep.yn2zoo(args.verb)

# Read in data
vaxes,vel = sep.read_file("in")
vel = vel.reshape(vaxes.n,order='F')
# Get axes of velocity model
nz = vaxes.n[0]; nx = vaxes.n[1]
oz = vaxes.n[0]; ox = vaxes.o[1]
dz = vaxes.d[0]; dx = vaxes.d[1]

# Loop over positions and put in anomalies
for ia in range(na):
  # Compute gaussian anomaly
  gauss = np.zeros([nz,nx])
  gauss[pzs[ia],pxs[ia]] = 1.0
  gauss = flt.gaussian_filter(gauss,sigma=rects[ia])/np.max(flt.gaussian_filter(gauss,sigma=rects[ia]))*scales[ia]
  # Add gaussian
  vel += gauss

# Write to file
sep.write_file("out",vaxes,vel)

