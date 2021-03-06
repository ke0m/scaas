%.o: %.cpp
	g++ -std=c++1y -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

default: scaas2dpy halfint trismooth slantstk adcig

laplacian10.o: laplacian10.ispc
	ispc --target=avx $< -o $@

scaas2dpy: scaas2d.o laplacian10.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g laplacian10.o scaas2d.o scaas2dpy.cpp -o ../scaas2dpy`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

halfint: halfint.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fft.o kiss_fftr.o halfint.o halfintpy.cpp -o ../halfint`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

trismooth: trismooth.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g trismooth.o trismoothpy.cpp -o ../trismoothkern`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

slantstk: slantstk.o halfint.o aainterp.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fftr.o kiss_fft.o halfint.o aainterp.o slantstk.o slantstkpy.cpp -o \
		../slantstk`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

adcig: adcig.o tan2ang.o slantstk.o halfint.o aainterp.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fftr.o kiss_fft.o halfint.o aainterp.o slantstk.o tan2ang.o \
		adcig.o adcigpy.cpp -o ../adcig`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

adcigkzkx: adcigkzkx.o ctrist.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g  ctrist.o adcigkzkx.o adcigkzkxpy.cpp -o ../adcigkzkx`${CONDA_PREFIX}/bin/python3-config --extension-suffix` -lfftw3f
