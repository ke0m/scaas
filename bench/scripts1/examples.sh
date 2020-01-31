#! /bin/bash

# Examples using the python scripts

# Model data
python scripts/ModelData.py -nsx 20 -dsx 20 -osx 40 -plotacq y src=srcmarm15.H vel=marmnopad.H out=marmdat.H

# Model data in parallel
python scripts/ModelDataParallel.py -nsx 20 -dsx 20 -osx 40 -plotacq y src=srcmarm15.H vel=marmnopad.H out=marmdat.H -nthreads 4
py35 scripts/ModelDataParallel.py -nsx 47 -dsx 10 -osx 0 -plotacq y src=srcmarm15.H vel=marmnopad.H out=marmdat.H -nthreads 24

# Model a wavefield
python scripts/ModelWfld.py -srcx 250 -srcz 0 -plotacq y src=srcmarm15.H vel=marmnopad.H out=wfld.H

# Gradient for a single shot
python scripts/ModelData.py -nsx 3 -dsx 100 -osx 150 -plotacq y src=srcmarm15.H vel=marmnopad.H out=marm3sht.H
python scripts/Gradient.py in=marm3sht.H src=srcmarm15wind.H vel=marmsmth50t.H out=marmgrad.H
python scripts/GradientDx.py in=marmdat.H src=srcmarm15.H vel=marmsmth50twind.H out=marmgrad.H -plotacq y
python scripts/GradientDt.py in=marmdat.H src=srcmarm15.H vel=marmsmth50twind.H out=marmgraddt.H -plotacq y

py35 scripts/GradientParallel.py in=marmdat.H src=srcmarm15.H vel=marmsmth50twind.H out=marmgrad.H -plotacq y -nthreads 24

py35 scripts/grad-func.py in=marmdat.H src=srcmarm15.H vel=marmsmth50twind.H out=marmgradtap.H -nthreads 24 -izt 55 -izb 70 -moddat=synmarmdat.H

Window n3=1 f3=1 < marm3sht.H > marmsht.H
Grey crowd1=0.37 labelfat=3 label1='Time (s)' label2='Receiver' title=' ' < marmsht.H labelrot=n out=shot.v > /dev/null

# FWI on Marmousi
py35 scripts/FWI_vel.py in=marmdat.H src=srcmarm15scale.H vel=marmsmth50twind.H out=fwimarm.H -ofn=marmofnpy.H -gmov=marmgmovpy.H -mmov=marmmmovpy.H -dmov=marmdmovpy.H -izt 55 -izb 70 -nthreads 24 -plotacq y -wtrials y -trim y -sidxs 5,20,30,40

# Gaussian example
Vel n1=100 o1=0.0 d1=10 n2=100 o2=0.0 d2=10 n3=1 o3=0 d3=1 vc=2000  > velback.H
py35 scripts/AddGaussians.py in=velback.H -scales 100,-100,100 -pzs 50,50,50 -pxs 20,50,80 out=vel3g.H
Window n1=3000 < srcmarm15.H > srcmarm15wind2.H
Scale rscale=400 < srcmarm15wind2.H > srcmarm15wind2scale.H
py35 scripts/ModelDataParallel.py -nsx 50 -dsx 2 -osx 0 -recz 99 -bx 50 -bz 50 -plotacq y src=srcmarm15wind2scale.H vel=vel3g.H out=trudat.H -nthreads 24 -plotacq y
py35 scripts/grad-func-tx.py in=trudat.H src=srcmarm15wind2scale.H vel=velback.H out=grad3g.H -nthreads 24 -moddat=syndat3g.H -iztt 50 -izbt 70 -iztb 140 -izbb 160

py35 scripts/FWI_vel-tx.py in=trudat.H src=srcmarm15wind2scale.H vel=velback.H out=fwi3g.H -ofn=tgofn.H -gmov=tggmov.H -mmov=tgmmov.H -dmov=tgdmov.H -iztt 50 -izbt 70 -iztb 140 -izbb 160 -nthreads 24 -plotacq y -wtrials y -trim y -sidxs 5,20,30,40

