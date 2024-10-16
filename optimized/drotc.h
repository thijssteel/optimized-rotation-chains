#ifndef DROTC_BLOCKS_H
#define DROTC_BLOCKS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

/**
 * @brief Apply a chain of k rotation sequences of length n to a block of A.
 * 
 * @param m     integer
 *              The number of rows in the block
 * 
 * @param n     integer
 *              The number of waves to apply
 * 
 * @param k     integer
 *              The length of each wave
 * 
 * @param Ap    m x (n+k) array of doubles
 *              The packed matrix to which the rotations are to be applied
 * 
 * @param ldap  integer
 *              The leading dimension of Ap, i.e. the distance between two rows.
 *              Note that because of the packing, this is only valid for multiples of MR.
 * 
 * @param C     n x k array of doubles
 *              The cosines of the rotations
 * 
 * @param ldc   integer
 *              The leading dimension of C, ldc >= n
 *              Note: C is assumed to be a rectangular block, not a trapezoidal block
 * 
 * @param S     n x k array of doubles
 *              The sines of the rotations
 * 
 * @param lds   integer
 *              The leading dimension of S, lds >= n
 *              Note: S is assumed to be a rectangular block, not a trapezoidal block
 */
void drotc_pipeline_block(int m, int n, int k, double *Ap, int ldap, const double *C, int ldc, const double *S, int lds);


void drotc(char side, char dir, bool startup, bool shutdown, int m, int n, int k, double *A, int lda, const double *C, int ldc, const double *S, int lds);

void drotc_prepacked(char side, char dir, bool startup, bool shutdown,
           int m, int n, int k, double *A_pack, int ldap,
           const double *C, int ldc, const double *S, int lds);

#ifdef __cplusplus
}
#endif


#endif