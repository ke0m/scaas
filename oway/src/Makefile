CPPFLAGS=-std=c++1y
CFLAGS=-O3 -fPIC -ffast-math -fopenmp -shared -Wall
INCS=-I. -I/home/joseph29/opt/FFTW/include
LDFLAGS=-L/home/joseph29/opt/FFTW/lib
LDLIBS=-lfftw3f

%.o: %.cpp
	g++ $(CPPFLAGS) $(CFLAGS) $(INCS) -c -o $@ $<

%.o: %.c
	gcc  $(CFLAGS) $(INCS) -c -o $@ $<

ssr3: ssr3.o
	g++ $(CPPFLAGS) $(CFLAGS) $(INCS) \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g ssr3.o ssr3py.cpp -o ../ssr3`${CONDA_PREFIX}/bin/python3-config --extension-suffix` \
		$(LDFLAGS) $(LDLIBS)

ssr3wrap: ssr3wrap.o ssr3.o
	g++ $(CPPFLAGS) $(CFLAGS) $(INCS) \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g ssr3.o ssr3wrap.o ssr3wrappy.cpp -o ../ssr3wrap`${CONDA_PREFIX}/bin/python3-config --extension-suffix` \
		$(LDFLAGS) $(LDLIBS)

mutter: mute.o
	g++ $(CPPFLAGS) $(CFLAGS) `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g mute.o mutepy.cpp -o ../mutter`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

costapkern: costaper.o
	g++ $(CPPFLAGS) $(CFLAGS) `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g costaper.o costaperpy.cpp -o ../costapkern`${CONDA_PREFIX}/bin/python3-config --extension-suffix`
