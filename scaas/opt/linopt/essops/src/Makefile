TCH_PREFIX=${CONDA_PREFIX}/lib/python3.7/site-packages/torch


%.o: %.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

%.o: %.c 
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

slowtfm: slow.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g slow.o slowpy.cpp -o ../slowtfm`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

slowtorch.o:
	g++ -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I${TCH_PREFIX}/include -I${TCH_PREFIX}/include/torch/csrc/api/include \
		-I${TCH_PREFIX}/include/TH -I${TCH_PREFIX}/include/THC -I${CONDA_PREFIX}/include/python3.7m -c slowtorch.cpp -o slowtorch.o \
		-DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=slowtorch -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11

slowtorch: slow.o slowtorch.o
	g++ -shared -L${CONDA_PREFIX}/lib -Wl,-rpath=${CONDA_PREFIX}/lib \
	 	slow.o slowtorch.o -o ../slowtorch.cpython-37m-x86_64-linux-gnu.so

