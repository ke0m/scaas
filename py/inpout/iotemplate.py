# IO template for SEP python programs
from __future__ import print_function
import sys, os, argparse, ConfigParser
import seppy
import numpy as np
import subprocess

# Parse the config file
conf_parser = argparse.ArgumentParser(add_help=False)
conf_parser.add_argument("-c", "--conf_file",
                         help="Specify config file", metavar="FILE")
args, remaining_argv = conf_parser.parse_known_args()
defaults = { 
    "verb": "n",
    "klean": "y",
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
parser.add_argument("in=",help="input file")
parser.add_argument("out=",help="output file")
# Required arguments
requiredNamed = parser.add_argument_group('required parameters')
requiredNamed.add_argument('-ex',help='A required argument',required=True,type=str)
# Optional arguments
parser.add_argument("-klean",help="Clean files ([y] or n)")
parser.add_argument("-verb",help="Verbosity flag (y or [n])")
args = parser.parse_args(remaining_argv)

verb  = args.verb
clean = args.klean
if(verb == "n"):
  verb = 0
else:
  verb = 1
if(clean == "n"):
  clean = 0
else:
  clean = 1

# Set up SEP
sep = seppy.sep(sys.argv)


# Clean up
if(clean):
  rm = " "
  if(verb): print(rm)
  sp = subprocess.check_call(rm,shell=True)

