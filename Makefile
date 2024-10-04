CC=gcc
CFLAGS=-march=native -Wall -O3 -g

CXX=g++
CXXFLAGS=-march=native -Wall -g

test: test/test_kernels_avx2

optimized/kernels_avx2_double.o: optimized/kernels_avx2_double.c
	$(CC) -c -o $@ $< $(CFLAGS)

optimized/kernels_avx2_single.o: optimized/kernels_avx2_single.c
	$(CC) -c -o $@ $< $(CFLAGS)

test/test_kernels_avx2: test/test_kernels_avx2.cpp optimized/kernels_avx2_double.o optimized/kernels_avx2_single.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I.

