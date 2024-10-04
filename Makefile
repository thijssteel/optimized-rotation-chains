CC=gcc
CFLAGS=-march=native -Wall -g -O3

CXX=g++
CXXFLAGS=-march=native -Wall -g

test: test/test_kernels_avx2 test/test_block_avx2

optimized/kernels_avx2_double.o: optimized/kernels_avx2_double.c
	$(CC) -c -o $@ $< $(CFLAGS)

optimized/kernels_avx2_single.o: optimized/kernels_avx2_single.c
	$(CC) -c -o $@ $< $(CFLAGS)

optimized/rotc_avx2_double.o: optimized/rotc_avx2_double.c optimized/kernels_avx2.h
	$(CC) -c -o $@ $< $(CFLAGS)

test/test_kernels_avx2: test/test_kernels_avx2.cpp optimized/kernels_avx2_double.o optimized/kernels_avx2_single.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I.

test/test_block_avx2: test/test_block_avx2.cpp optimized/kernels_avx2_double.o optimized/kernels_avx2_single.o optimized/rotc_avx2_double.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I.

