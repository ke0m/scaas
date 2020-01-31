"""
A test script for my implementation of a linear 
conjugate direction solver

@author: Joseph Jennings
@version: 2019.11.28
"""
from __future__ import print_function
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import opt.linopt as opt
from linfunctions import *
import matplotlib.pyplot as plt 

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = { 
    "verb": "y",
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
parser.add_argument("-verb",help="Verbosity flag ([y] or n)",type=str)
args = parser.parse_args(remaining_argv)

# Get command line arguments
sep = seppy.sep(sys.argv)
verb = sep.yn2zoo(args.verb)

# Solver inputs
A = np.array([[1,1],[2,-1]]) # operator
b = np.array([4,5])          # data
x = np.zeros(2)              # model

g  = np.zeros(2)             # gradient
r  = np.zeros(2)             # residual
dr = np.zeros(2)             # data-space gradient

# Create working dictonary
w = {}
w['s']  = np.zeros(2,dtype='float32')
w['ss'] = np.zeros(2,dtype='float32')

# Create info dictionary
idict = {}
idict['f']     = 0.0
idict['iflag'] = 1
idict['iter']  = 0
idict['niter'] = 20
idict['rtol']  = 0.0001
idict['verb']  = verb

# Keep track of the iterations
itercheck = 0
icall = 0

while(True):
  # Compute objective function and necessary vectors
  f = matvec(A,x,b,g,r,dr)

  # Update model via conjugate directions
  opt.cdstep(f,x,g,r,dr,w,idict)

  # Check if a new iterate was found
  if(itercheck != idict['iter'] or idict['iflag'] == 0):
    itercheck = idict['iter']
    #TODO: write the diagnostics

  icall += 1
  if(idict['iflag'] <= 0 or icall > 500): break

print(x)
