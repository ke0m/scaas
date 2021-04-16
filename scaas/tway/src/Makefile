%.o: %.cpp
	g++ -std=c++1y -O3 -fPIC -ffast-math -fopenmp -shared -Wall \
		-I. -I../../external_cpp/ -c -fPIC -o $@ $<

INSTALL_DIR=..

default: scaas2dpy

laplacian10.o: laplacian10.ispc
	ispc --target=avx $< -o $@

scaas2dpy: scaas2d.o laplacian10.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g laplacian10.o scaas2d.o scaas2dpy.cpp -o \
		$(INSTALL_DIR)/scaas2dpy`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

