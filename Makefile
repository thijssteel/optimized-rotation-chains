CC=gcc
CFLAGS=-Wall -g -O3

CXX=g++
CXXFLAGS=-Wall -g -O3

test: test/test_kernels_ref test/test_kernels_avx test/test_kernels_avx2

optimized/drotc_kernels_ref.o: optimized/drotc_kernels_ref.c optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS)

optimized/drotc_kernels_avx.o: optimized/drotc_kernels_avx.c optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS) -march=ivybridge

optimized/drotc_kernels_avx2.o: optimized/drotc_kernels_avx2.c optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS) -march=skylake

test/test_kernels_ref: test/test_kernels.cpp optimized/drotc_kernels_ref.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I.

test/test_kernels_avx: test/test_kernels.cpp optimized/drotc_kernels_avx.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=ivybridge

test/test_kernels_avx2: test/test_kernels.cpp optimized/drotc_kernels_avx2.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=skylake