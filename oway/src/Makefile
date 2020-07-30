%.o: %.cpp
	g++ -std=c++1y -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

ssr3: ssr3.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g ssr3.o ssr3py.cpp -o ../ssr3`${CONDA_PREFIX}/bin/python3-config --extension-suffix` -lfftw3f

ssr3wrap: ssr3wrap.o ssr3.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g ssr3.o ssr3wrap.o ssr3wrappy.cpp -o ../ssr3wrap`${CONDA_PREFIX}/bin/python3-config --extension-suffix` -lfftw3f

ssr3_kiss: ssr3_kiss.o kiss_fft.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fft.o ssr3_kiss.o ssr3py_kiss.cpp -o ../ssr3_kiss`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

mutter: mute.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g mute.o mutepy.cpp -o ../mutter`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

costapkern: costaper.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g costaper.o costaperpy.cpp -o ../costapkern`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

ompfunc: ompfunc.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g ompfunc.o ompfuncpy.cpp -o ../ompfunc`${CONDA_PREFIX}/bin/python3-config --extension-suffix`
