%.o: %.cpp
	g++ -std=c++1y -O3 -fPIC -ffast-math -fopenmp -shared -Wall \
		-I. -I../../external_cpp -c -fPIC -o $@ $<

%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

INSTALL_DIR=..

default: halfint slantstk adcig adcigkzkx

halfint: halfint.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fft.o kiss_fftr.o halfint.o halfintpy.cpp -o \
		$(INSTALL_DIR)/halfint`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

slantstk: slantstk.o halfint.o aainterp.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fftr.o kiss_fft.o halfint.o aainterp.o slantstk.o slantstkpy.cpp -o \
		$(INSTALL_DIR)/slantstk`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

adcig: adcig.o tan2ang.o slantstk.o halfint.o aainterp.o kiss_fft.o kiss_fftr.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fftr.o kiss_fft.o halfint.o aainterp.o slantstk.o tan2ang.o \
		adcig.o adcigpy.cpp -o \
		$(INSTALL_DIR)/adcig`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

adcigkzkx: adcigkzkx.o ctrist.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g  ctrist.o adcigkzkx.o adcigkzkxpy.cpp -o \
		$(INSTALL_DIR)/adcigkzkx`${CONDA_PREFIX}/bin/python3-config --extension-suffix` \
		-lfftw3f
