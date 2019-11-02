PYPATH=/opt/anaconda/anaconda3/envs/py35/

laplacian10.o: laplacian10.ispc
	/opt/ispc-v1.9.2-linux/ispc --target=avx $< -o $@

laplacianold.o: laplacianFWDISPC.ispc #laplacian10.ispc
	/opt/ispc-v1.9.2-linux/ispc --target=avx $< -o $@

scaas2d.o: scaas2d.cpp
	g++ -O3 -fPIC -ffast-math -std=c++1y -shared -Wall -I${PYPATH}/include/python3.5m \
		-Wl,-rpath=${PYTPATH}/lib -L${PYPATH}/lib -lpython3.5m -c $< -o $@

#XXX: The -shared is necessary for linking
scaas2dpy: scaas2d.o laplacian10.o laplacianold.o
	g++ -O3 -fPIC -ffast-math -std=c++1y -shared -Wall `python3 -m pybind11 --includes` \
		-g laplacian10.o scaas2d.o scaas2dpy.cpp -o ../scaas2dpy`python3-config --extension-suffix`

testimshow: testimshow.cpp
	g++ testimshow.cpp -I/opt/anaconda/anaconda3/envs/py35/include/python3.5m/ \
		-Wl,-rpath=/opt/anaconda/anaconda3/envs/py35/lib \
		-L/opt/anaconda/anaconda3/envs/py35/lib -lpython3.5m -o testimshow -std=c++11

testbasic: testbasic.cpp
	g++ testbasic.cpp -I/opt/anaconda/anaconda3/envs/py35/include/python3.5m/ \
		-Wl,-rpath=/opt/anaconda/anaconda3/envs/py35/lib \
		-L/opt/anaconda/anaconda3/envs/py35/lib -lpython3.5m -o $@ -std=c++11

laplaciantest: laplaciantest.cpp laplacian10.o
	g++ $< -I/opt/anaconda/anaconda3/envs/py35/include/python3.5m/ \
		-Wl,-rpath=/opt/anaconda/anaconda3/envs/py35/lib \
		-L/opt/anaconda/anaconda3/envs/py35/lib -lpython3.5m laplacian10.o -o $@ -std=c++11

laplacianoldtest: laplacianoldtest.cpp laplacianold.o
	g++ $< -I/opt/anaconda/anaconda3/envs/py35/include/python3.5m/ \
		-Wl,-rpath=/opt/anaconda/anaconda3/envs/py35/lib \
		-L/opt/anaconda/anaconda3/envs/py35/lib -lpython3.5m laplacianold.o -o $@ -std=c++11

