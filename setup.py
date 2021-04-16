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
    # Modify Makefile to
    cmdtway = "cd ./scaas/scaas/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    print("Executing {}".format(cmdtway))
    subprocess.check_call(cmdtway, shell=True)
    cmdoway = "cd ./scaas/oway/src && make INSTALL_DIR=%s" % (
        site.getsitepackages()[0])
    print("Executing {}".format(cmdoway))
    subprocess.check_call(cmdoway, shell=True)
    build_py.run(self)


class Develop(develop):
  """Custom build for building PyBind11 modules in development mode"""

  def run(self):
    cmdtway = "cd ./scaas/scaas/src && make"
    print("Executing {}".format(cmdtway))
    subprocess.check_call(cmdtway, shell=True)
    cmdoway = "cd ./scaas/oway/src && make"
    print("Executing {}".format(cmdoway))
    subprocess.check_call(cmdoway, shell=True)
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
