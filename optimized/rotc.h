#ifndef ROTC_H_
#define ROTC_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void drotc_pipeline_block_right(int m, int n, int k, double *A, int lda, double *Ap, const double *C, int ldc, const double *S, int lds, double *P);

void drotc(char side, char dir, bool startup, bool shutdown, int m, int n, int k, double *A, int lda, const double *C, int ldc, const double *S, int lds);

void srotc(char side, char dir, bool startup, bool shutdown, int m, int n, int k, float *A, int lda, const float *C, int ldc, const float *S, int lds);

#ifdef __cplusplus
}
#endif

#endif