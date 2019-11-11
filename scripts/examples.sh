#! /bin/bash

# Examples using the python scripts

# Model data
python scripts/ModelData.py -nsx 20 -dsx 20 -osx 40 -plotacq y src=srcmarm15.H vel=marmnopad.H out=marmdat.H

# Model data in parallel
python scripts/ModelDataParallel.py -nsx 20 -dsx 20 -osx 40 -plotacq y src=srcmarm15.H vel=marmnopad.H out=marmdat.H -nthreads 4

# Model a wavefield
python scripts/ModelWfld.py -srcx 250 -srcz 0 -plotacq y src=srcmarm15.H vel=marmnopad.H out=wfld.H
