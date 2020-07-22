NPPATH=${CONDA_PREFIX}/lib/python3.5/site-packages/numpy/core

%.o: %.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

ssr3.o: ssr3.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall -I${CONDA_PREFIX}/include/python3.5m \
		-I${NPPATH}/include -Wl,-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lpython3.5m -c $< -o $@

ssr3: ssr3.o kiss_fft.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fft.o ssr3.o ssr3py.cpp -o ../ssr3`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

ssr3_fftw: ssr3_fftw.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g ssr3_fftw.o ssr3py_fftw.cpp -o ../ssr3_fftw`${CONDA_PREFIX}/bin/python3-config --extension-suffix` -lfftw3f

mutter: mute.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g mute.o mutepy.cpp -o ../mutter`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

costapkern: costaper.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g costaper.o costaperpy.cpp -o ../costapkern`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

