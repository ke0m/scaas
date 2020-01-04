"""
Self-doc goes here

@author:
@version:
"""
import sys, os, argparse, configparser
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
parser.add_argument("-verb",help="Verbosity flag (y or [n])",type=str)
args = parser.parse_args(remaining_argv)

# Set up SEP
sep = seppy.sep(sys.argv)

# Get command line arguments
verb  = sep.yn2zoo(args.verb)
