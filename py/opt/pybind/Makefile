PYPATH=/opt/anaconda/anaconda3/envs/py35/

%.o: %.cpp
	g++ -O3 -fPIC -ffast-math -fopenmp -shared -Wall -I. -c -fPIC -o $@ $<

optpy: lbfgs_c.o nlcg.o lsearch.o steepest.o la.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall `${PYPATH}/bin/python3 -m pybind11 --includes` \
		-g la.o lsearch.o steepest.o nlcg.o lbfgs_c.o optpy.cpp -o ../optpy`${PYPATH}/bin/python3-config --extension-suffix`

test1c.x: test1c.o lbfgs_c.o steepest.o nlcg.o la.o lsearch.o functions.o
	g++ -O3 -fPIC -ffast-math -fopenmp -Wall -std=c++11 $^ -o $@

