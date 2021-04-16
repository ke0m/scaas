%.o: %.cpp
	g++ -std=c++1y -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

INSTALL_DIR=..

default: trismooth mutter costapkern

trismooth: trismooth.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g trismooth.o trismoothpy.cpp -o $(INSTALL_DIR)/trismoothkern`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

mutter: mute.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g mute.o mutepy.cpp -o ../mutter`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

costapkern: costaper.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g costaper.o costaperpy.cpp -o ../costapkern`${CONDA_PREFIX}/bin/python3-config --extension-suffix`


