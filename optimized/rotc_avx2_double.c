#include "immintrin.h"

#include "rotc.h"
#include "kernels_avx2.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

/**
 * Pack a block of m rows of A into A_packed, with m <= mr
 */
void pack_A_small(int m, int n, const double *A, int lda, double *A_packed)
{
    const int mr = 12;

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i ++)
        {
            A_packed[i + j * mr] = A[i + j * lda];
        }
    }
}

/**
 * Unpack a block of m rows of A_packed into A, with m <= mr
 */
void unpack_A_small(int m, int n, const double *A_packed, double *A, int lda)
{
    const int mr = 12;

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i ++)
        {
            A[i + j * lda] = A_packed[i + j * mr];
        }
    }
}

/**
 * Pack a block of kr columns of C and S into P
 */
void pack_C_and_S_mr(int n, const double *C, int ldc, const double *S, int lds, double *P)
{
    const int kr = 3;

    for (int j = 0; j < n; j++)
    {
        for (int p = 0; p < kr; p++)
        {
            P[2 * (j * kr + p)] = C[j + p * (ldc - 1)];
            P[2 * (j * kr + p) + 1] = S[j + p * (lds - 1)];
        }
    }
}

/**
 * Pack one column of C and S into P
 * This specialized for the case kr = 1 which handles remainders.
 */
void pack_C_and_S_small(int n, const double *C, int ldc, const double *S, int lds, double *P)
{
    for (int j = 0; j < n; j++)
    {
        P[2 * j] = C[j];
        P[2 * j + 1] = S[j];
    }
}

/**
 * @brief Apply a trapezoidal sequence of plane rotations to a matrix
 *
 * Specifically, the routine applies n waves of length k to the m-by-n+k matrix
 * A from the right.
 *
 * @param m   number of rows of A
 * @param n   number of waves to apply
 * @param k   length of each wave
 * @param A   m x (n+k) matrix
 *            The matrix to which the rotations are to be applied
 *            A is assumed to be in column major format
 * @param lda number of elements between consecutive columns of A
 * @param Ap  Space for the packed A matrix
 * @param C   (n+k) x k array of cosines
 *            C is assumed to be in column major format
 * @param ldc number of elements between consecutive columns of C
 * @param S   (n+k) x k array of sines
 *            S is assumed to be in column major format
 * @param lds number of elements between consecutive columns of S
 * @param P   Space for the packed C and S matrices
 */
void drotc_pipeline_block_right(int m, int n, int k, double *A, int lda, double *Ap, const double *C, int ldc, const double *S, int lds, double *P)
{
    const int mr = 12;
    const int kr = 3;

    int ldap = n + k;
    int ldp = 2 * n;

    for (int i = 0; i < m; i += mr)
    {
        int m2 = min(m - i, mr);
        pack_A_small(m2, n + k, &A[i], lda, &Ap[i * ldap]);

        int p = 0;
        int g = k - 1;
        for (; p + (kr - 1) < k; p += kr, g -= kr)
        {
            if(i == 0) pack_C_and_S_mr(n, &C[g + p * ldc], ldc, &S[g + p * lds], lds, &P[p * ldp]);
            drotc_kernel_12xnx3(n, &Ap[i * ldap + (g - (kr - 1)) * mr], &P[p * ldp]);
        }
        for (; p < k; p += 1, g -= 1)
        {
            if(i == 0) pack_C_and_S_small(n, &C[g + p * ldc], ldc, &S[g + p * lds], lds, &P[p * ldp]);
            drotc_kernel_12xnx1(n, &Ap[i * ldap + g * mr], &P[p * ldp]);
        }

        unpack_A_small(m2, n + k, &Ap[i * ldap], &A[i], lda);
    }
}

// void drotc(char side, char dir, bool startup, bool shutdown, int m, int n, int k, double *A, int lda, const double *C, int ldc, const double *S, int lds)
// {
//     const int mr = 12;
//     const int kr = 3;

//     // Make sure that kb and mb are multiples of kr and mr respectively
//     const int nb = 216;
//     const int kb = 60;
//     const int mb = 960;

//     const int m_pack = (min(m, mb) + mr - 1) / mr * mr;

//     double *A_pack =
//         (double *)aligned_alloc(64, m_pack * n * sizeof(double));

//     for (int ib = 0; ib < m; ib += mb)
//     {
//         int m2 = min(m - ib, mb);

//         pack_A(m2, n, &A[ib], lda, A_pack);

//         for (int pb = 0; pb < k; pb += kb)
//         {
//             int k2 = min(k - pb, kb);

//             // Startup
//             rot_sequence_startup(m2, k2, A_pack, n, &C[pb * ldc], ldc,
//                                  &S[pb * ldc], lds);

//             // Pipeline phase
//             for (int jb = 0; jb < n - k2; jb += nb)
//             {
//                 int n2 = min(n - k2 - jb, nb);

//                 rot_sequence_block(m2, n2, k2, &A_pack[jb * mr], n,
//                                    &C[jb + pb * ldc], ldc,
//                                    &S[jb + pb * ldc], lds);
//             }
//             // Shutdown
//             rot_sequence_shutdown(m2, k2, &A_pack[(n - k2) * mr], n,
//                                   &C[n - k2 + pb * ldc], ldc,
//                                   &S[n - k2 + pb * ldc], lds);
//         }

//         unpack_A(m2, n, &A[ib], lda, A_pack);
//     }

//     free(A_pack);
// }