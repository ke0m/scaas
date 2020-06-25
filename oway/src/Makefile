%.o: %.cpp
	g++ -std=c++1y -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

#halfint: ssr3.o kiss_fft.o
#	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
#		-g kiss_fft.o ssr3.o  halfintpy.cpp -o ../halfint`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

