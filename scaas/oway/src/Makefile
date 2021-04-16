%.o: %.cpp
	g++ -std=c++1y -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

INSTALL_DIR=..

default: ssr3

ssr3: ssr3.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g ssr3.o ssr3py.cpp -o ../ssr3`${CONDA_PREFIX}/bin/python3-config --extension-suffix` -lfftw3f

ssr3wrap: ssr3wrap.o ssr3.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g ssr3.o ssr3wrap.o ssr3wrappy.cpp -o ../ssr3wrap`${CONDA_PREFIX}/bin/python3-config --extension-suffix` -lfftw3f

