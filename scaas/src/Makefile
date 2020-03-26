NPPATH=${CONDA_PREFIX}/lib/python3.5/site-packages/numpy/core

%.o: %.cpp
	g++ -O3 -std=c++11 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

laplacian10.o: laplacian10.ispc
	${ISPC}/ispc --target=avx $< -o $@

laplacianold.o: laplacianFWDISPC.ispc #laplacian10.ispc
	${ISPC}/ispc --target=avx $< -o $@

scaas2d.o: scaas2d.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall -I${CONDA_PREFIX}/include/python3.5m \
		-I${NPPATH}/include -Wl,-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lpython3.5m -c $< -o $@

#XXX: The -shared is necessary for linking
scaas2dpy: scaas2d.o laplacian10.o laplacianold.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g laplacian10.o scaas2d.o scaas2dpy.cpp -o ../scaas2dpy`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

halfint: halfint.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fft.o kiss_fftr.o halfint.o halfintpy.cpp -o ../halfint`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

trismooth: trismooth.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g trismooth.o trismoothpy.cpp -o ../trismoothkern`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

trismooth.o: trismooth.cpp
	g++ -c trismooth.cpp -I${CONDA_PREFIX}/include/python3.5m/ \
		-Wl,-rpath=${CONDA_PREFIX}/lib -fPIC \
		-L${CONDA_PREFIX}/lib -lpython3.5m -o $@ -std=c++11

#adcig.o: adcig.cpp
# g++ -c adcig.cpp -I${CONDA_PREFIX}/include/python3.5m/ \
#   -Wl,-rpath=${CONDA_PREFIX}/lib -fPIC \
#   -L${CONDA_PREFIX}/lib -lpython3.5m -o $@ -std=c++11

tan2ang.o: tan2ang.cpp
	g++ -c tan2ang.cpp -I${CONDA_PREFIX}/include/python3.5m/ \
    -Wl,-rpath=${CONDA_PREFIX}/lib -fPIC \
    -L${CONDA_PREFIX}/lib -lpython3.5m -o $@ -std=c++11

slantstk: slantstk.o halfint.o aainterp.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
    -g kiss_fftr.o kiss_fft.o halfint.o aainterp.o slantstk.o slantstkpy.cpp -o ../slantstk`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

adcig: adcig.o tan2ang.o slantstk.o halfint.o aainterp.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fftr.o kiss_fft.o halfint.o aainterp.o slantstk.o tan2ang.o \
		adcig.o adcigpy.cpp -o ../adcig`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

testimshow: testimshow.cpp
	g++ testimshow.cpp -I${CONDA_PREFIX}/include/python3.5m/ \
		-Wl,-rpath=${CONDA_PREFIX}/lib \
		-L${CONDA_PREFIX}/lib -lpython3.5m -o testimshow -std=c++11

testbasic: testbasic.cpp
	g++ testbasic.cpp -I${CONDA_PREFIX}/include/python3.5m/ \
		-Wl,-rpath=${CONDA_PREFIX}/lib \
		-L${CONDA_PREFIX}/lib -lpython3.5m -o $@ -std=c++11

laplaciantest: laplaciantest.cpp laplacian10.o
	g++ $< -I${CONDA_PREFIX}/include/python3.5m/ \
		-Wl,-rpath=${CONDA_PREFIX}/lib \
		-L${CONDA_PREFIX}/lib -lpython3.5m laplacian10.o -o $@ -std=c++11

laplacianoldtest: laplacianoldtest.cpp laplacianold.o
	g++ $< -I${CONDA_PREFIX}/include/python3.5m/ \
		-Wl,-rpath=${CONDA_PREFIX}/lib \
		-L${CONDA_PREFIX}/lib -lpython3.5m laplacianold.o -o $@ -std=c++11

progressbar.x: progressbar.cpp
	g++ -std=c++1y -fopenmp $< -o $@

