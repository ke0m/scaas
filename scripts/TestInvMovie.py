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
import opt.optqc as optqc
from functions import *
import matplotlib.pyplot as plt 

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = { 
    "wtrials": "n",
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
parser.add_argument("-wtrials",help="Write trial results (y or [n])")
args = parser.parse_args(remaining_argv)

# Set up IO
sep = seppy.sep(sys.argv)
wtrials = sep.yn2zoo(args.wtrials)

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

# Create the optqc objects
dia  = optqc.optqc(sep,"-ofn",ofaxes,"-mmov","-gmov",mgaxes)
# Write the first iteration
dia.output(f,x,g)

diat = None
if(wtrials):
  diat = optqc.optqc(sep,"-ofn",ofaxes,"-mmov","-gmov",mgaxes,trials=True)
  # Write the first iteration
  diat.output(f,x,g)

# Keep track of iterations
itercheck = 0

while(True):
  # User supplied objective function and gradient evaluation function
  f[0] = rosenbrock(x,g)

  # Write trial steps if requested
  if(wtrials):
    diat.output(f,x,g)

  # Call solver
  opt.lbfgs(n,m,x,f,g,diagco,diag,w,iflag,isave,dsave)

  # Check if a new iterate was found
  if(itercheck != isave[4] or iflag[0] == 0):
    # Update the iteration number
    itercheck = isave[4]
    # Write the new iterate, gradient and objective function
    dia.output(f,x,g)

  icall += 1
  if(iflag[0] <= 0 or icall > 500): break

# Print the output x
print(x)
