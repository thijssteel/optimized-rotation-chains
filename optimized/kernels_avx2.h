#ifndef ROTC_KERNELS_AVX2_H_
#define ROTC_KERNELS_AVX2_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Kernels apply n wavefronts of size kr to a mr rows of A.
 * 
 * A should be packed in column major format.
 * P should be packed in the following order:
 * [c00, s00, c01, s01, ..., c0(kr-1), s0(kr-1), c10, s10, ..., c(n-1)(kr-1), s(n-1)(kr-1)]
 * 
 * The name of the function indicates the parameters:
 * rotc_kernel_mrxnxkr
 */

void drotc_kernel_12xnx3(int n, double * A, const double * P);

void drotc_kernel_12xnx1(int n, double * A, const double * P);

void srotc_kernel_24xnx3(int n, float * A, const float * P);

void srotc_kernel_24xnx1(int n, float * A, const float * P);

#ifdef __cplusplus
}
#endif

#endif // ROTC_KERNELS_AVX2_H_