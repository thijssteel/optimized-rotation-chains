CC=gcc
CFLAGS=-Wall -g -O3

CXX=g++
CXXFLAGS=-Wall -g -O3

LAPACK_LIBS=-m64  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

test: test/test_kernels_ref test/test_kernels_avx test/test_kernels_avx2

# Reference implementation

reference/drotc.o: reference/drotc.f90
	gfortran -c -o $@ $<

reference/srotc.o: reference/srotc.f90
	gfortran -c -o $@ $<

reference/zrotc.o: reference/zrotc.f90
	gfortran -c -o $@ $<

reference/crotc.o: reference/crotc.f90
	gfortran -c -o $@ $<

reference/dzrotc.o: reference/dzrotc.f90
	gfortran -c -o $@ $<

reference/scrotc.o: reference/scrotc.f90
	gfortran -c -o $@ $<

# Actual implementation

optimized/drotc_kernels_ref.o: optimized/drotc_kernels_ref.c optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS)

optimized/drotc_kernels_avx.o: optimized/drotc_kernels_avx.c optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS) -march=ivybridge

optimized/drotc_kernels_avx2.o: optimized/drotc_kernels_avx2.c optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS) -march=skylake

optimized/drotc_kernels_avx512.o: optimized/drotc_kernels_avx512.c optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS) -march=icelake-server

optimized/drotc_ref.o: optimized/drotc.c optimized/drotc.h optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS)

optimized/drotc_avx.o: optimized/drotc.c optimized/drotc.h optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS) -march=ivybridge

optimized/drotc_avx2.o: optimized/drotc.c optimized/drotc.h optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS) -march=skylake

optimized/drotc_avx512.o: optimized/drotc.c optimized/drotc.h optimized/drotc_kernels.h optimized/drotc_params.h
	$(CC) -c -o $@ $< $(CFLAGS) -march=icelake-server

# Tests

test/test_kernels_ref: test/test_kernels.cpp optimized/drotc_kernels_ref.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I.

test/test_kernels_avx: test/test_kernels.cpp optimized/drotc_kernels_avx.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=ivybridge

test/test_kernels_avx2: test/test_kernels.cpp optimized/drotc_kernels_avx2.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=skylake

test/test_kernels_avx512: test/test_kernels.cpp optimized/drotc_kernels_avx512.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=icelake-server

test/test_drotc_ref: test/test_drotc.cpp optimized/drotc_kernels_ref.o optimized/drotc_ref.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I.

test/test_drotc_avx: test/test_drotc.cpp optimized/drotc_kernels_avx.o optimized/drotc_avx.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=ivybridge

test/test_drotc_avx2: test/test_drotc.cpp optimized/drotc_kernels_avx2.o optimized/drotc_avx2.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=skylake

test/test_drotc_avx512: test/test_drotc.cpp optimized/drotc_kernels_avx512.o optimized/drotc_avx512.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=icelake-server

test/profile_drotc_ref: test/profile_drotc.cpp optimized/drotc_kernels_ref.o optimized/drotc_ref.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I.

test/profile_drotc_avx: test/profile_drotc.cpp optimized/drotc_kernels_avx.o optimized/drotc_avx.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=ivybridge

test/profile_drotc_avx2: test/profile_drotc.cpp optimized/drotc_kernels_avx2.o optimized/drotc_avx2.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=skylake

test/profile_drotc_avx512: test/profile_drotc.cpp optimized/drotc_kernels_avx512.o optimized/drotc_avx512.o
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=icelake-server

test/profile_gemm: test/profile_gemm.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS) -lstdc++ -I. -march=native $(LAPACK_LIBS)
