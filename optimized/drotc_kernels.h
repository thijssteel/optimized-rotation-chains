#ifndef DROTC_KERNELS_H
#define DROTC_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Kernel that applies n wavefronts of kr rotations to an mr x (n+kr) matrix A.
 *        A should be packed, i.e. A should be in column-major format with lda = mr.
 *        C and S should be in column major format with the same leading dimension.
 * 
 * @param n     integer
 *              The number of wavefronts to apply
 * 
 * @param A     mr x (n+kr) array of doubles
 *              The matrix to which the rotations are to be applied
 * 
 * @param C     n x kr array of doubles
 *              The cosines of the rotations
 * 
 * @param ldc   integer
 *              The leading dimension of C, ldc >= n
 *              Note: C is assumed to be a rectangular block, not a trapezoidal block
 * 
 * @param S     n x kr array of doubles
 *              The sines of the rotations
 *  
 * @param lds  integer
 *             The leading dimension of S, lds >= n
 *             Note: S is assumed to be a rectangular block, not a trapezoidal block
 */
void drotc_kernel_mrxnxkr(int n, double * A, const double * C, int ldc, const double * S, int lds);

/**
 * @brief Kernel that applies n rotations to an mr x (n+1) matrix A.
 *        A should be packed, i.e. A should be in column-major format with lda = mr.
 *        This is a smaller kernel that is less efficient but is needed for edge cases.
 * 
 * @param n     integer
 *              The number of wavefronts to apply
 * 
 * @param A     mr x (n+1) array of doubles
 *              The matrix to which the rotations are to be applied
 * 
 * @param C     size n array of doubles
 *              The cosines of the rotations
 * 
 * @param S     size n array of doubles
 *              The sines of the rotations
 */
void drotc_kernel_mrxnx1(int n, double * A, const double * C, const double * S);

/**
 * @brief Pack an m x n block of the matrix A into a packed matrix Ap.
 * 
 * @param m    integer
 *             The number of rows in the block
 * 
 * @param n    integer
 *             The number of columns in the block
 * 
 * @param A    m x n array of doubles
 *             The matrix to pack
 * 
 * @param lda  integer
 *             The leading dimension of A
 * 
 * @param Ap   mp x n array of doubles
 *             The packed matrix
 */
void drotc_pack_A(int m, int n, const double * A, int lda, double * Ap);

/**
 * @brief Unpack an m x n block of the matrix A from a packed matrix Ap.
 * 
 * @param m    integer
 *             The number of rows in the block
 * 
 * @param n    integer
 *             The number of columns in the block
 * 
 * @param Ap   mp x n array of doubles
 *             The packed matrix
 * 
 * @param A    m x n array of doubles
 *             The matrix to unpack
 * 
 * @param lda  integer
 *             The leading dimension of A
 */
void drotc_unpack_A(int m, int n, const double * Ap, double * A, int lda);


#ifdef __cplusplus
}
#endif

#endif