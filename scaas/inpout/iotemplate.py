"""
Self-doc goes here

@author:
@version:
"""
import sys, os, argparse, configparser
import inpout.seppy as seppy
import numpy as np
import subprocess

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = {
    "verb": "n",
    }
if args.conf_file:
  config = configparser.ConfigParser()
  config.read([args.conf_file])
  defaults = dict(config.items("defaults"))

# Parse the other arguments
# Don't surpress add_help here so it will handle -h
parser = argparse.ArgumentParser(parents=[conf_parser],description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)

# Set defaults
parser.set_defaults(**defaults)

# Input files
ioArgs = parser.add_argument_group('Inputs and outputs')
ioArgs.add_argument("in=",help="input file",type=str)
ioArgs.add_argument("out=",help="output file",type=str)
# Required arguments
requiredNamed = parser.add_argument_group('required parameters')
requiredNamed.add_argument('-ex',help='A required argument',required=True,type=str)
# Optional arguments
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
# Enables required arguments in config file
for action in parser._actions:
  if(action.dest in defaults):
    action.required = False
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
