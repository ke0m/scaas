"""
A test script for Huy's LBFGS, Non-linear conjugate gradient
and steepest descent solvers

Minimizes the following functions:
  1) Quadratic
  2) Six hump camelback function (https://www.sfu.ca/~ssurjano/camel6.html)
  3) Rosenbrock function (https://www.sfu.ca/~ssurjano/rosen.html)
  4) Powell function (https://www.sfu.ca/~ssurjano/powell.html)
  5) Trid function (https://www.sfu.ca/~ssurjano/trid.html)
  6) Variant of the Rosenbrock function

It does so using one of three gradient descent solvers:
  1) Steepest descent
  2) Non-linear conjugate gradient
  3) Limited memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS)

@author: Joseph Jennings
@version: 2019.11.22
"""
from __future__ import print_function
import sys, os, argparse, configparser
import numpy as np
import opt.nlopt as opt
from functions import *
import matplotlib.pyplot as plt 

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = { 
    "function": "quad",
    "solver": "lbfgs",
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
# Quality check
parser.add_argument("-function",help="[quad],camel6,rbrock,powell,trid,rbrockvariant",type=str)
parser.add_argument("-solver",help="[lbfgs],nlcg,steepest",type=str)
args = parser.parse_args(remaining_argv)

# Get command line arguments
func = args.function
solv = args.solver

# Set the size of the x (the minimizer/unknown)
n = 0
if(func == "quad" or func == "camel6"):
  n = 2
elif(func  == "trid"):
  n = 50
elif(func == "rbrockvariant"):
  n = 25
else:
  n = 100

# Solver inputs
# m is the number of remembered gradients used to build approximated Hessian
m = np.zeros(1,dtype='int32') + 5
iflag = np.zeros(1,dtype='int32')
diagco = False; icall = 0

# Objective function value and unknown
f = np.zeros(1,dtype='float32')
x = np.zeros(n,dtype='float32')

# Create initial guesses for minimizer
if(func == "quad" or func == "camel6"):
  x[0] = -1;
  x[1] = -0.5;
elif(func == "rbrock"):
  for i in range(n): x[i] = 2
elif(func == "powell"):
  for i in range(0,n,4):
    x[i+0] =  3.0
    x[i+1] = -1.0
    x[i+3] =  1.0

# Gradient
g = np.zeros(n,dtype='float32')
diag = np.zeros(n,dtype='float32')

# Create working array 
if(solv == "steepest"):
  w = np.zeros(n,dtype='float32')
elif(solv == "nlcg"):
  w = np.zeros(2*n+1,dtype='float32')
else:
  w = np.zeros(n*(2*m[0]+1)+2*m[0],dtype='float32')

# Saved state variables
isave = np.zeros(8,dtype='int32')
dsave = np.zeros(14,dtype='float32')

while(True):
  # User supplied objective function and gradient evaluation function
  if(func == "quad"):
    f[0] = quad(x,g)
  elif(func == "camel6"):
    f[0] = camel6(x,g)
  elif(func == "rbrock"):
    f[0] = rosenbrock(x,g)
  elif(func == "powell"):
    f[0] = powell(x,g)
  elif(func == "trid"):
    f[0] = trid(x,g)
  elif(func == "rbrockvariant"):
    f[0] = rosenbrock1(x,g)

  # Call solver
  if(solv == "steepest"):
    opt.steepest(n,x,f,g,diag,w,iflag,isave,dsave)
  elif(solv == "nlcg"):
    opt.nlcg(n,x,f,g,diag,w,iflag,isave,dsave)
  else:
    opt.lbfgs(n,m,x,f,g,diagco,diag,w,iflag,isave,dsave)

  icall += 1
  if(iflag[0] <= 0 or icall > 200): break

# Print the output x
print(x)

