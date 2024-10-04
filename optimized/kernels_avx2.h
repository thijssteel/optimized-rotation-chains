#ifndef ROTC_KERNELS_AVX2_H_
#define ROTC_KERNELS_AVX2_H_

#ifdef __cplusplus
extern "C" {
#endif

void drotc_kernel_12xnx3(int n, double * A, const double * P);

void drotc_kernel_12xnx1(int n, double * A, const double * P);

void srotc_kernel_24xnx3(int n, float * A, const float * P);

void srotc_kernel_24xnx1(int n, float * A, const float * P);

#ifdef __cplusplus
}
#endif

#endif // ROTC_KERNELS_AVX2_H_