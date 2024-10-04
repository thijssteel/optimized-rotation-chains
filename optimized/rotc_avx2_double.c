#include "immintrin.h"

#include "rotc.h"
#include "kernels_avx2.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

// Pack the m x n matrix A into A_packed
// A_packed stores mr values contiguously before moving to the next column
void pack_A(int m, int n, const double* A, int lda, double* A_packed)
{
    const int mr = 12;

    int ib = 0;
    for (; ib + (mr - 1) < m; ib += mr) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < mr; i += 4) {
                __m256d a = _mm256_loadu_pd(&A[i + ib + j * lda]);
                _mm256_store_pd(&A_packed[ib * n + j * mr + i], a);
            }
        }
    }
    if (ib < m) {
        // Less than mr rows remaining in A
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m - ib; i++) {
                A_packed[ib * n + j * mr + i] = A[i + ib + j * lda];
            }
        }
    }
}

void drotc(char side, char dir, bool startup, bool shutdown, int m, int n, int k, double *A, int lda, const double *C, int ldc, const double *S, int lds)
{
    const int mr = 12;
    const int kr = 3;

    // Make sure that kb and mb are multiples of kr and mr respectively
    const int nb = 216;
    const int kb = 60;
    const int mb = 960;

    const int m_pack = (min(m, mb) + mr - 1) / mr * mr;

    double *A_pack =
        (double *)aligned_alloc(64, m_pack * n * sizeof(double));

    for (int ib = 0; ib < m; ib += mb)
    {
        int m2 = min(m - ib, mb);

        pack_A(m2, n, &A[ib], lda, A_pack);

        for (int pb = 0; pb < k; pb += kb)
        {
            int k2 = min(k - pb, kb);

            // Startup
            rot_sequence_startup(m2, k2, A_pack, n, &C[pb * ldc], ldc,
                                 &S[pb * ldc], lds);

            // Pipeline phase
            for (int jb = 0; jb < n - k2; jb += nb)
            {
                int n2 = min(n - k2 - jb, nb);

                rot_sequence_block(m2, n2, k2, &A_pack[jb * mr], n,
                                   &C[jb + pb * ldc], ldc,
                                   &S[jb + pb * ldc], lds);
            }
            // Shutdown
            rot_sequence_shutdown(m2, k2, &A_pack[(n - k2) * mr], n,
                                  &C[n - k2 + pb * ldc], ldc,
                                  &S[n - k2 + pb * ldc], lds);
        }

        unpack_A(m2, n, &A[ib], lda, A_pack);
    }

    free(A_pack);
}