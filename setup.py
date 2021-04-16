from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
import subprocess
import site

VERSION = '0.0.1'
DESCRIPTION = """A package for imaging and tomographic inversion
                 of reflection seismic data"""
LONG_DESCRIPTION = """A package for imaging and tomographic inversion
                     of reflection seismic data. Has options to use both the
                     second-order acoustic wave equation as well as the one-way
                     wave equation"""


class Build(build_py):
  """Custom build for building PyBind11 modules"""

  def run(self):
    # Get the external libs
    cmdsub = 'git submodule init && git submodule update'
    subprocess.check_call(cmdsub, shell=True)
    # Two-way
    cmdtway = "cd ./scaas/tway/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    subprocess.check_call(cmdtway, shell=True)
    # One-way
    cmdoway = "cd ./scaas/oway/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    subprocess.check_call(cmdoway, shell=True)
    # Filter
    cmdfltr = "cd ./scaas/filter/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    subprocess.check_call(cmdfltr, shell=True)
    # Off2ang
    cmdof2an = "cd ./scaas/off2ang/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    subprocess.check_call(cmdof2an, shell=True)
    build_py.run(self)


class Develop(develop):
  """Custom build for building PyBind11 modules in development mode"""

  def run(self):
    # Get the external libs
    cmdsub = 'git submodule init && git submodule update'
    subprocess.check_call(cmdsub, shell=True)
    # Two-way
    cmdtway = "cd ./scaas/tway/src && make"
    subprocess.check_call(cmdtway, shell=True)
    # One-way
    cmdoway = "cd ./scaas/oway/src && make"
    subprocess.check_call(cmdoway, shell=True)
    # Filter
    cmdfltr = "cd ./scaas/filter/src && make"
    subprocess.check_call(cmdfltr, shell=True)
    # Off2ang
    cmdof2an = "cd ./scaas/off2ang/src && make"
    subprocess.check_call(cmdof2an, shell=True)
    develop.run(self)


# Setting up
setup(
    name="scaas",
    version=VERSION,
    author="Joseph Jennings",
    author_email="<joseph29@sep.stanford.edu>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    cmdclass={
        'build_py': Build,
        'develop': Develop,
    },
    keywords=['seismic', 'imaging', 'inversion', 'tomography'],
    classifiers=[
        "Intended Audience :: Seismic processing/imaging",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux ",
    ],
)
