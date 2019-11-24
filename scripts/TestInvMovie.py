"""
Tests the non-linear optimization of the Rosenbrock 
function using the LBFGS algorithm

Writes the current model, gradient and objective
function value at each iteration

@author: Joseph Jennings
@version: 2019.11.23
"""
import sys, os, argparse, configparser
import numpy as np
import inpout.seppy as seppy
import opt.optpy as opt
from functions import *
import matplotlib.pyplot as plt 

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = { 
    }   
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.set_defaults(**defaults)
# Get output inversion movie files
parser.add_argument("-mmov",help="Output model movie")
parser.add_argument("-gmov",help="Output gradient movie")
parser.add_argument("-ofn",help="Output objective function")
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep(sys.argv)
# Get the names of the files
mmov = sep.get_fname("-mmov")
gmov = sep.get_fname("-gmov")
ofn  = sep.get_fname("-ofn")

# Size of model
n = 100

# Solver inputs
# m is the number of remembered gradients used to build approximated Hessian
m = np.zeros(1,dtype='int32') + 5
iflag = np.zeros(1,dtype='int32')
diagco = False; icall = 0

# Objective function value and unknown
f = np.zeros(1,dtype='float32')
x = np.zeros(n,dtype='float32')

# Intial guess for minimizer
for i in range(n): x[i] = 2

# Gradient
g = np.zeros(n,dtype='float32')
diag = np.zeros(n,dtype='float32')

# Create working array 
w = np.zeros(n*(2*m[0]+1)+2*m[0],dtype='float32')

# Saved state variables
isave = np.zeros(8,dtype='int32')
dsave = np.zeros(14,dtype='float32')

# Create axes
mgaxes = seppy.axes([n],[0.0],[1.0])
ofaxes = seppy.axes([1],[0.0],[1.0])

# Write first iteration
if(mmov != None):
  sep.write_file("-mmov",mgaxes,x)
if(gmov != None):
  sep.write_file("-gmov",mgaxes,g)
if(ofn != None):
  sep.write_file("-ofn",ofaxes,f)
  ofaxes.ndims = 0

# Keep track of iterations
itercheck = 0

while(True):
  # User supplied objective function and gradient evaluation function
  f[0] = rosenbrock(x,g)

  # Call solver
  opt.lbfgs(n,m,x,f,g,diagco,diag,w,iflag,isave,dsave)

  # Check if a new iterate was found
  if(itercheck != isave[4] or iflag[0] == 0):
    # Update the iteration number
    itercheck = isave[4]
    if(iflag[0] == 0): itercheck += 1
    # Write the new iterate, gradient and objective function
    if(mmov != None):
      sep.append_to_movie("-mmov",mgaxes,x,itercheck+1)
    if(gmov != None):
      sep.append_to_movie("-gmov",mgaxes,g,itercheck+1)
    if(ofn != None):
      sep.append_to_movie("-ofn",ofaxes,f,itercheck+1)

  icall += 1
  if(iflag[0] <= 0 or icall > 200): break

# Print the output x
print(x)
