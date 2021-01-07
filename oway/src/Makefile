NPPATH=${CONDA_PREFIX}/lib/python3.7/site-packages/numpy/core

%.o: %.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -march=native -mtune=native -shared -Wall -I. -c -fPIC -o $@ $<

%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -march=native -mtune=native -shared -Wall -I. -c -fPIC -o $@ $<

ssr3.o: ssr3.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -march=native -mtune=native -std=c++1y -shared -Wall \
		-I${CONDA_PREFIX}/include/python3.5m \
		-I${NPPATH}/include -Wl,-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lpython3.5m -c $< -o $@

ssr3_kiss.o: ssr3_kiss.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall -I${CONDA_PREFIX}/include/python3.5m \
		-I${NPPATH}/include -Wl,-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lpython3.5m -c $< -o $@

ssr3: ssr3.o
	g++ -O3 -fPIC -ffast-math -fopenmp -march=native -mtune=native -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g ssr3.o ssr3py.cpp -o ../ssr3`${CONDA_PREFIX}/bin/python3-config --extension-suffix` -lfftw3f

ssr3wrap: ssr3wrap.o ssr3.o
	g++ -O3 -fPIC -ffast-math -fopenmp -march=native -mtune=native -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
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

testoway2d.x: profile/testoway2d.cpp ssr3.o
	g++ -O3 -fPIC -ffast-math -fopenmp -march=native -mtune=native -std=c++1y -Wall \
		ssr3.o $< -o $@ -lfftw3f

testoway2d_bench.x: profile/testoway2d_bench.cpp ssr3.o
	g++ -O3 -fopenmp -march=native -mtune=native -std=c++1y -Wall -I/opt/benchmark/include \
		-L/opt/benchmark/build/src ssr3.o $< -o $@ -lfftw3f -lbenchmark -lbenchmark_main -lpthread

testoway2d_plot.x: profile/testoway2d_plot.cpp ssr3.o
	g++ -O3 -fPIC -ffast-math -fopenmp -march=native -mtune=native -std=c++1y -Wall \
		-I${CONDA_PREFIX}/include/python3.5m \
		-I${NPPATH}/include -Wl,-rpath=${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib ssr3.o \
		$< -o $@ -lfftw3f -lpython3.5m

quantile_test.x: profile/quantile_test.cpp
	g++ -O3 $< -o $@
