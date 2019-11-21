#! /bin/bash

# Examples using the python scripts

# Model data
python scripts/ModelData.py -nsx 20 -dsx 20 -osx 40 -plotacq y src=srcmarm15.H vel=marmnopad.H out=marmdat.H

# Model data in parallel
python scripts/ModelDataParallel.py -nsx 20 -dsx 20 -osx 40 -plotacq y src=srcmarm15.H vel=marmnopad.H out=marmdat.H -nthreads 4

# Model a wavefield
python scripts/ModelWfld.py -srcx 250 -srcz 0 -plotacq y src=srcmarm15.H vel=marmnopad.H out=wfld.H

# Gradient for a single shot
python scripts/ModelData.py -nsx 3 -dsx 100 -osx 150 -plotacq y src=srcmarm15.H vel=marmnopad.H out=marm3sht.H
python scripts/Gradient.py in=marm3sht.H src=srcmarm15wind.H vel=marmsmth50t.H out=marmgrad.H
python scripts/GradientDx.py in=marmdat.H src=srcmarm15.H vel=marmsmth50twind.H out=marmgrad.H -plotacq y
python scripts/GradientDt.py in=marmdat.H src=srcmarm15.H vel=marmsmth50twind.H out=marmgraddt.H -plotacq y


Window n3=1 f3=1 < marm3sht.H > marmsht.H
Grey crowd1=0.37 labelfat=3 label1='Time (s)' label2='Receiver' title=' ' < marmsht.H labelrot=n out=shot.v > /dev/null
